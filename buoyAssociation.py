import cv2
import sys
import os
import numpy as np
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import threading
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DistanceEstimator'))
from DistanceEstimator import DistanceEstimator

from utility.Transformations import ECEF2LatLng, T_ECEF_Ship, LatLng2ECEF, haversineDist
from utility.GeoData import GetGeoData
from utility.Rendering import RenderAssociations
from utility.LivePlotting import LivePlots
from boxmot import ByteTrack
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class BuoyAssociation():    # 3.75
    def __init__(self, focal_length=3.75, pixel_size=0.00155, img_sz=[1920, 1080]):
        self.focal_length = focal_length        # focal length of camera in mm
        self.scale_factor = 1 / (2*pixel_size)  # scale factor of camera -> pixel size in mm
        self.image_size = img_sz
        self.conv_thresh = 0.25  # used for NMS and BoxMOT -> Detections below this thresh won't be considered
        self.distanceEstimator = DistanceEstimator(img_size = 1024, conv_thresh = self.conv_thresh, iou_thresh = 0.2)    # load Yolov7 with Distance Module, conv & iou_thresh for NMS
        self.BuoyCoordinates = GetGeoData(tile_size=0.02) # load BuoyData from GeoJson
        self.imu_data = None
        self.RenderObj = None   # render Instance
        self.track_buffer = 60              # after exceeding thresh (frames count) a lost will be reassigned new ID 
        self.MOT = self.initBoxMOT()        # Multi Object Tracker Instance
        self.ma_storage = {}    # dict to store moving averages
        self.matching_confidence = {}
        self.matching_confidence_plotting = {}

    def test(self, images_dir, labels_dir, imu_dir, plots=True):
        #function tests performance of BuoyAssociation on Labeled Set of Images including BuoyGT

        if plots:
            plots_folder = self.create_run_directory(path="detections/")
        self.imu_data = self.getIMUData(imu_dir)   # load IMU data

        for image in os.listdir(images_dir):   
            image_path = os.path.join(images_dir, image) 
            print("image: ", image_path)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read {image_path}")
                continue
            idx = int(os.path.basename(image_path).replace(".png", "")) -1  # frame name to imu index -> frames start with 1, IMU with 0
            pred, pred_dict = self.getPredictions(img, idx, moving_average=False)

            labels = self.getLabelsData(labels_dir, image_path)
            labels_wo_gps = [x[:-1] for x in labels]
            buoyLabels = [x[-1] for x in labels]
            pred_dict["buoys_labeled"] = buoyLabels      

            if plots:
                self.distanceEstimator.plot_inference_results(pred, img, name=os.path.basename(image_path), 
                                                                folder=plots_folder, labelsData=labels_wo_gps)
                self.plot_Predictions(pred_dict, name = os.path.basename(image_path).replace(".png", ""), 
                                            folder=plots_folder)

    def getPredictions(self, img, frame_id, moving_average=True):
        """Function runs inferense, returns Yolov7 preds concatenated with bearing to Objects & ID of BoxMOT
        Prediction dict contains ship pose and predicted buoy positions (in lat,lng)
        Args:
            Img: pixel array
            frame_id: current frame ID
        Returns:
            preds: (n,8) tensor [xyxy, conf, cls, dist, angle, ID]
            pred_dict dict containing "ship": [lat, lng, heading] and "buoy_predictions": [[lat1,lng1], ...]
        """

        pred = self.distanceEstimator(img)
        pred = self.getAnglesOfIncidence(pred)  # get pixel center coordinates of BBs and compute lateral angle
        
        # BoxMOT tracking on predictions
        preds_boxmot_format = pred[:,0:6]    # preds need to be in format [xyxy, conf, classID]
        preds_boxmot_format = np.array(preds_boxmot_format)
        res = self.MOT.update(preds_boxmot_format, img)   # res --> M X (x, y, x, y, id, conf, cls, ind)
        res = torch.from_numpy(res)
        ids = -1*torch.ones(size=(pred.size()[0], 1))   # default case -1
        if len(res) > 0:
            ids[res[:,-1].to(torch.int32), 0] = res[:, 4].to(torch.float)
        pred = torch.cat((pred, ids), dim=-1)   # pred = [N x [xyxy,conf,classID,dist,angle,ID]]

        if moving_average:
            pred = self.moving_average(pred, frame_id, method='EMA')

        pred_dict = self.BuoyLocationPred(frame_id, pred)   # compute buoy predictions
        return pred, pred_dict
        
    def getAnglesOfIncidence(self, preds):
        """Computes angle of deviation between optical axis of cam and object in x (horizontal direction)

        Returns:
            prediction tensor concatenated with angle: on (n,8) tensor per image [xyxy, conf, cls, dist, angle]
        """

        x_center = (preds[:,0] + preds[:,2]) / 2    # convert left and right corner x coords to center coord x
        u_0 = self.image_size[0] / 2
        x = (x_center - u_0) / self.scale_factor
        alpha = -1*torch.arctan((x)/self.focal_length).unsqueeze(1)
        preds = torch.cat((preds, alpha), dim=-1)
        return preds
    
    def moving_average(self, preds, frameID, method='UEMA'):
        """Computes moving average over dist predictions specified by method
        Args:
            preds: prediction tensor with ID at last pos [N x [xyxy,conf,class,dist,angle,ID]]
            method: EMA (Exp Moving Average), UEMA (Unbiased Exp Moving Average), WMA (Window Moving average) 

        Returns:
            prediction tensor with dist values modified in place -> [N x [xyxy,conf,class,dist,angle,ID]]
        """

        # EMA / UEMA Parameters
        alpha = 0.9
        # WMA: Window Size
        w_sz = 10

        # clean up dict entries of tracks that exceed max frame threshold
        for k in list(self.ma_storage.keys()):
            if (frameID - self.ma_storage[k]['frameID']) > self.track_buffer:
                del self.ma_storage[k]

        for pred in preds:
            id = int(pred[-1])
            if id == -1:
                continue    # if no id has been assigned to BB, keep dist prediction
            if id in self.ma_storage:
                self.ma_storage[id]['frameID'] = frameID
                if method == 'EMA':
                    self.ma_storage[id]['G'] = alpha * self.ma_storage[id]['G'] + (1-alpha) * pred[6] # compute EMA
                    pred[6] = self.ma_storage[id]['G']
                elif method == 'UEMA':
                    self.ma_storage[id]['S'] = alpha * self.ma_storage[id]['S'] + pred[6] # compute EMA
                    i = self.ma_storage[id]['n']
                    pred[6] = (1-alpha) / (1 - alpha**(i+1)) * self.ma_storage[id]['S']
                    self.ma_storage[id]['n'] += 1
                elif method == 'WMA':
                    self.ma_storage[id]['data'].append(pred[6])
                    if len(self.ma_storage[id]) > w_sz:
                        self.ma_storage[id]['data'].pop(0)
                    pred[6] = sum(self.ma_storage[id]) / len(self.ma_storage[id])
                else:
                    raise ValueError("Method {method} not defined")
            else:
                if method == 'EMA':
                    self.ma_storage[id] = {'G': pred[6], 'frameID': frameID}
                elif method == 'UEMA':
                    self.ma_storage[id] = {'n': 1, 'S': pred[6], 'frameID': frameID}
                elif method == 'WMA':
                    self.ma_storage[id] = {'data': [pred[6]], 'frameID': frameID}
                else:
                    raise ValueError("Method {method} not defined")

        return preds

    
    def BuoyLocationPred(self, frame_id, preds):
        """for each BB prediction function computes the Buoy Location based on Dist & Angle of the tensor
        Args:
            frame_id: ID of current frame
            preds: prediction tensor of yolov7 (N,8) -> [Nx[xyxy, conf, cls, dist, angle]]
        Returns:
            Dict{"ship:"[lat,lng,heading], "buoy_prediction":[[lat1,lng1],[lat2,lng2]]}
        """

        latCam = self.imu_data[frame_id][3]
        lngCam = self.imu_data[frame_id][4]
        heading = self.imu_data[frame_id][2]

        # trasformation:    latlng to ecef, ecef to enu, enu to ship
        x, y, z = LatLng2ECEF(latCam, lngCam)  # ship coords in ECEF
        ECEF_T_Ship = T_ECEF_Ship(x,y,z,heading)   # transformation matrix between ship and ecef

        # compute 2d points (x,y) in ship cs, (z=0, since all objects are on water surface)
        buoysX = (torch.cos(preds[:,7]) * preds[:,6]).tolist()
        buoysY = (torch.sin(preds[:,7]) * preds[:,6]).tolist()
        buoy_preds = list(zip(buoysX, buoysY))

        # transform buoyCoords to lat lng
        buoysLatLng = []
        for buoy in buoy_preds:
            p = ECEF_T_Ship @ np.array([buoy[0], buoy[1], 0, 1])    # buoy coords in ecef
            lat, lng, alt = ECEF2LatLng(p[0],p[1],p[2])
            buoysLatLng.append((lat, lng))

        return {"buoy_predictions": buoysLatLng, "ship": [latCam, lngCam, heading]}

    def getLabelsData(self, labels_dir, image_path):
        labelspath = os.path.join(labels_dir, os.path.basename(image_path) + ".json")
        if os.path.exists(labelspath):
            return self.distanceEstimator.LabelsJSONFormat(labelspath)
        else:
            print(f"LablesFile not found: {labelspath}")
            return []
        
    def getIMUData(self, path):
        # function returns IMU data as list

        if os.path.isfile(path):
            result = []
            with open(path, 'r') as f:
                data = f.readlines()
                for line in data:
                    content = line.split(",")
                    line = [float(x) for x in content]
                    result.append(line)
        else:
            files = os.listdir(path)
            filename = [f for f in files if f.endswith('.txt')][0]
            path = os.path.join(path, filename)
            result = []
            with open(path, 'r') as f:
                data = f.readlines()
                for line in data:
                    content = line.split(",")
                    line = [float(x) for x in content]
                    result.append(line)
            if len(result) == 0:
                print("No IMU data found, check path: {path}")
        return result
    
    def initBoxMOT(self):
        return ByteTrack(
            track_thresh=2 * self.conv_thresh,      # threshold for detection confidence -> seperates BBs into high and low confidence
            match_thresh=0.99,                  # matching thresh -> controls max dist allowed between tracklets & detections for a match
            track_buffer=self.track_buffer      # number of frames to keep a track alive after it was last detected
        )

    def plot_Predictions(self, data, name, folder):
        # transfoms buoy data to ship cs and plots them

        lat, lon, heading = data["ship"]
        x,y,z = LatLng2ECEF(lat, lon)
        ECEF_T_Ship = T_ECEF_Ship(x,y,z,heading)

        buoy_preds = []
        for lat, lng in data["buoy_predictions"]:
            x,y,z = LatLng2ECEF(lat, lng)
            p_buoy = np.linalg.pinv(ECEF_T_Ship) @ np.array([x,y,z,1])  # buoy pred in ship CS
            #p_buoy = self.getRotationMatrix(np.pi/2, 'z') @ p_buoy[0:3]
            buoy_preds.append(p_buoy[:2])

        buoy_label = []
        if "buoys_labeled" in data:
            for lat, lng in data["buoys_labeled"]:
                x,y,z = LatLng2ECEF(lat, lng)
                p_buoy = np.linalg.pinv(ECEF_T_Ship) @ np.array([x,y,z,1])  # buoy pred in ship CS
                #p_buoy = self.getRotationMatrix(np.pi/2, 'z') @ p_buoy[0:3]
                buoy_label.append(p_buoy[:2])

        # plot ship
        # Define arrow position and direction in the XY plane
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # Define arrow position and direction in the XY plane
        scaling = 30
        coords = np.asarray([[1,0,0], [-2,1, 0], [-1,0,0], [-2,-1,0], [1,0,0]])
        coords *= scaling

        poly = Poly3DCollection([coords], color=[(0,0,0.9)], edgecolor='k')
        ax.add_collection3d(poly)

        # Customize the view
        ax.set_xlim(-100, 700)
        ax.set_ylim(-400, 400)
        ax.set_zlim(0, 1)
        #ax.set_zlim(-1, 1)  # Keep it flat in the Z-axis for an XY view

        if len(buoy_preds) > 0:
            buoy_preds = np.asarray(buoy_preds)
            ax.plot3D(buoy_preds[:,0], buoy_preds[:,1], np.zeros(len(buoy_preds)), 'o', color = 'red')
            for pred in buoy_preds:
                ax.plot([0, pred[0]], [0,pred[1]],zs=[0,0], color='grey', linestyle='dashed')
        if len(buoy_label) > 0:
            buoy_label = np.asarray(buoy_label)
            ax.plot3D(buoy_label[:,0], buoy_label[:,1], 'o', color = 'green')
            for label in buoy_label:
                ax.plot([0, label[0]], [0, label[1]],zs=[0,0], color='grey', linestyle='dashed')
        # Optional: Labels for clarity
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=60, azim=-180)  # Set view to make it look like a 2D XY plane

        plt.draw()
        plt.savefig(os.path.join(folder, name+".pdf"), dpi=300)
        plt.close()
    
    def plot_Predictions_to_map(self, data, zoom=1.5, name="buoyPredictions", folder=""):
        # func expects a dict containing predicted buoys and ship as key and the lat lon pairs as values

        ship_lat, ship_lon, heading = data["ship"]
        
        # Create figure with Cartopy projection
        fig, ax = plt.subplots(figsize=(10, 10),
                            subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([ship_lon - zoom, ship_lon + zoom, ship_lat - zoom, ship_lat + zoom])

        # Add map features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.OCEAN, color='lightblue')
        
        # Plot buoy predictions in red
        buoypreds = data["buoy_predictions"]
        for lat, lon in buoypreds:
            ax.plot(lon, lat, 'ro', markersize=8, transform=ccrs.PlateCarree(), label="Buoy Predictions")

        # Plot buoys in green if they exist
        if "buoys_labeled" in data:
            buoys = data["buoys_labeled"]
            if buoys:
                for lat, lon in buoys:
                    ax.plot(lon, lat, 'go', markersize=8, transform=ccrs.PlateCarree(), label="Buoys")

        # Plot ship in blue with heading
        ax.plot(ship_lon, ship_lat, 'b^', markersize=12, transform=ccrs.PlateCarree(), label="Ship")
        # Add legend and labels
        plt.legend(loc="upper right")
        plt.title("Ship and Buoy Positions")
        path = os.path.join(folder, name+"_buoys.pdf")
        plt.savefig(path)

    def getColor(self, color_dict, frameID, id):
        # returns colordict with new entry & color added for ID. Removes old entries

        color_table = [(255/255, 255/255, 0, 1),
                       (102/255, 0, 102/255, 1),
                       (0, 255/255, 255/255, 1),
                       (255/255, 153/255, 255/255, 1),
                       (153/255, 102/255, 51/255, 1),
                       (255/255,153/155, 0, 1),
                       (224/255, 224/255, 224/255, 1),
                       (128/255, 128/255, 0, 1)
                       ]

        # clean up color_dict by removing id,color pairs that are old
        for k in list(color_dict.keys()):
            if (frameID - color_dict[k]['frame']) > self.track_buffer or k<0:
                del color_dict[k]
 
        # add new id with color to dict   
        clr = (90/255, 90/255, 90/255, 1)    # default color if all other colors are already taken    
        for color in color_table:
            if color not in [color_dict[k]["color"] for k in color_dict]:
                clr = color
                break
        color_dict[id] = {'color': clr, 'frame':frameID}

        return color_dict

    def create_run_directory(self, base_name="run", path=""):
        i = 0
        while True:
            folder_name = f"{base_name}{i if i > 0 else ''}"
            if not os.path.exists(os.path.join(path, folder_name)):
                path_to_folder = os.path.join(path, folder_name)
                os.makedirs(path_to_folder)
                print(f"Created directory: {path_to_folder} to store plots")
                return path_to_folder
            i += 1

    def displayFPS(self, frame, prev_frame_time):
        # function displays FPS on frame

        font = cv2.FONT_HERSHEY_SIMPLEX 
        new_frame_time = time.time() 
        fps= 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps) 
        fps = str(fps) 
        cv2.putText(frame, fps, (10, 20), font, 0.5, (50, 50, 50)) 
        return new_frame_time
    
    def Coords2Hash(self, coords):
        # takes a tuple of lat, lng coordinates and returns a unique hash key as string
        return str(coords[0])+str(coords[1])
    
    def computeMatchingConf(self, id, buoyGT, color): 
        # for each gt buoy compute confidence of all predictions ids in the past

        conf_multiplier = 1 / 15
        max_conf = 10
        k = self.Coords2Hash(buoyGT)
        if k not in self.matching_confidence:
            self.matching_confidence[k] = defaultdict(float)

        self.matching_confidence[k][id] += conf_multiplier  # increase multiplier of a with 
        self.matching_confidence[k][id] = min(max_conf, self.matching_confidence[k][id])

    def decayMatchingConf(self):
        # function decays matching conf of all entries in self.matching_confidence

        conf_decay = 1 / (2*15) 
        for k in self.matching_confidence:
            for id in self.matching_confidence[k]:
                self.matching_confidence[k][id] -= conf_decay
                if self.matching_confidence[k][id] < 0:
                    self.matching_confidence[k][id] = 0
    

    def matching(self, preds, buoys_chart):
        """function computes bipartite matching between chart buoys and predictions
        Args:
            preds: list of buoy predict coords of the form [[lat1,lng1],[lat2,lng2]]
            buoy_chart: list of buoy gt coords of the form [[lat1,lng1],[lat2,lng2]]
        Returns:
            list of turples of matches indices [(a,b),(c,d)] where a is index of gt and b of pred
        """

        G = []  # cost matrix
        for buoy in buoys_chart:
            edges = list(map(lambda x: int(haversineDist(*x, *buoy)), preds))
            G.append(edges)

        G = np.asarray(G)
        row_ind, col_ind = linear_sum_assignment(G)     # run hungarian algorithm
        result = [(a,b) for a,b in zip(row_ind, col_ind)]   # create tuples of matched indices
        return result

    def getNearbyBuoys(self, ship_pose, buoyCoords, fov_with_padding=100, dist_thresh=1200, nearby_thresh=50):
        """function selects nearby gt buoys (relative to ship pos) from a list containing buoy Coordinates
        Args:
            ship_pose: list of form [lat,lng,heading]
            buoyCoords: list of form [[lat1,lng1], [lat2, lng2], ...]
            fov_with_padding: fov of camera plus additional padding to account for inaccuracies
            dist_thresh: only buoys inside this threshold will be considered
            nearby_thresh: all buoys inside this thresh will pe added, even if outside of fov
        Returns:
            list of turples of matches indices [(a,b),(c,d)] where a is index of gt and b of pred
        """

        selected_buoys = []
        # compute transformation matrix from ecef to ship cs
        x, y, z = LatLng2ECEF(ship_pose[0], ship_pose[1])  # ship coords in ECEF
        Ship_T_ECEF = np.linalg.pinv(T_ECEF_Ship(x,y,z,ship_pose[2]))   # transformation matrix between ship and ecef

        for buoy in buoyCoords:
            lat = buoy["geometry"]["coordinates"][1]
            lng = buoy["geometry"]["coordinates"][0]
            x,y,z = LatLng2ECEF(lat, lng)
            pos_bouy = Ship_T_ECEF @ np.array([x,y,z,1]) # transform buoys from ecef to ship cs
            bearing = np.arctan2(pos_bouy[1],pos_bouy[0])   # compute bearing of buoy
            dist_to_ship = haversineDist(lat, lng, ship_pose[0], ship_pose[1])  # compute dist to ship
            if abs(bearing) <= fov_with_padding and dist_to_ship <= dist_thresh:
                # include buoys that are within fov+padding and inside maxdist
                selected_buoys.append((lat, lng))
            elif dist_to_ship <= nearby_thresh:
                # also include nearby buoys not inside FOV
                selected_buoys.append((lat, lng))

        return list(set(selected_buoys))

    def filterPreds(self, shipPose, preds, buoysGT, dist_thresh_far = 0.2, dist_thresh_close = 0.35):
        """ function filters predictions based on nearest neighbour search & thresholding
        thrshold is based on relative distance from pred to ship and distinguished between far (>150m) and close (<=150m)
        Args:
            shipPose:       [lat,lng,heading]
            preds:          list of buoy preds in lat lng format [lat,lng],...]
            buoysGT:        list of ground truth buoys in lat lng format [[lat,lng],...]
            dist_thresh:    threshold (relative w.r.t. closest dist to gt buoys / dist to ship)
        Returns:
            filtered_preds: filtered list of buoy predictions in lat lng format [[lat,lng],...]
            index_dict: matches original indices of prediction vector to new indices of filtered preds 
        """

        filtered_preds = []
        index_dict = {}
        for i, pred in enumerate(preds):
            distances = list(map(lambda x: haversineDist(*pred, *x), buoysGT))
            closestGT = np.argmin(distances)
            closest_dist = min(distances) # closest dist to gt buoys
            dist_to_ship = haversineDist(*pred, *shipPose[:2])  # dist between prediciton and ship
            dist_gt_ship = haversineDist(*buoysGT[closestGT], *shipPose[:2])    # dist between gt buoy pos and ship
            dist_thresh = dist_thresh_far if dist_gt_ship > 150 else dist_thresh_close
            if closest_dist / dist_to_ship <= dist_thresh:
                filtered_preds.append(pred)
                index_dict[len(filtered_preds)-1] = i
        return filtered_preds, index_dict

    def video(self, video_path, imu_path, rendering=False):
        # run buoy association on video

        if not rendering:
           self.processVideo(video_path, imu_path, rendering) 
        else:
            # initialize Rendering Framework with data
            lock = threading.Lock()
            self.RenderObj = RenderAssociations(lock)
            self.imu_data = self.getIMUData(imu_path)
            lat_start = self.imu_data[0][3]
            lng_start = self.imu_data[0][4]
            heading_start = self.imu_data[0][2]
            self.RenderObj.initTransformations(lat_start, lng_start, heading_start) # initialize Transformation Matrices with pos & heading of first frame
            # start thread to run video processing 
            processing_thread = threading.Thread(target=self.processVideo, args=(video_path, imu_path, rendering,lock,), daemon=True)
            processing_thread.start()
            # start rendering
            self.RenderObj.run()

    def initMatchingConfPlots(self):
        plt.ion()
        fig = plt.figure()
        return fig

    def updatePlots(self, fig, color_dict):
        for ax in fig.axes:
            ax.remove()

        data = self.matching_confidence_plotting

        size = len([k for k in data])
        if size == 0:
            return 
        gs = matplotlib.gridspec.GridSpec(1,size)

        for i,k in enumerate(data):
            ax = fig.add_subplot(gs[0, i])
            for id in data[k]:
                ax.plot(data[k][id]['data'], color=data[k][id]['color'])
            ax.set_title("BuoyGT")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Metric")
        fig.canvas.draw()
        fig.canvas.flush_events()

    def prepareData(self, color_dict):
        for k in self.matching_confidence:
            if k not in self.matching_confidence_plotting:
                self.matching_confidence_plotting[k] = {}

            for id in self.matching_confidence[k]:
                if id in self.matching_confidence_plotting[k]:
                    self.matching_confidence_plotting[k][id]['data'].append(self.matching_confidence[k][id])
                else:
                    color = (0.2, 0.2, 0.2, 1) if id < 0 else color_dict[id]
                    self.matching_confidence_plotting[k][id] = {'data': [self.matching_confidence[k][id]], 'color': color}
            
    def processVideo(self, video_path, imu_path, rendering, lock=None):     
        # function computes predictions, and performs matching for each frame of video

        # load IMU data
        self.imu_data = self.getIMUData(imu_path)

        # load geodata
        lat_start = self.imu_data[0][3]
        lng_start = self.imu_data[0][4]
        buoyCoords = self.BuoyCoordinates.getBuoyLocations(lat_start, lng_start)
        newBuoyCoords = threading.Event()   # event that new data has arrived from thread
        results_list = []
        #self.BuoyCoordinates.plotBuoyLocations(buoyCoords)

        fig = self.initMatchingConfPlots()
        # refreshPlots = threading.Event()
        # lp = LivePlots(self, refreshPlots)


        cap = cv2.VideoCapture(video_path)
        current_time = time.time()
        color_assignment = {}

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_id = 0
        paused = False
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                
                # get predictions for frame
                pred, pred_dict = self.getPredictions(frame, frame_id, moving_average=True)

                # draw BBs on frame
                self.distanceEstimator.drawBoundingBoxes(frame, pred, color=(1,0,0), conf_thresh=self.conv_thresh)

                # check if new buoy coords have been set by thread
                if newBuoyCoords.is_set():
                    newBuoyCoords.clear()   # clear event flag
                    buoyCoords = results_list   # copy results list
                    results_list = []   # clear results list
                
                # check if buoydata needs to be reloaded
                refresh = self.BuoyCoordinates.checkForRefresh(pred_dict["ship"][0], pred_dict["ship"][1])
                if refresh:
                    # load new buoycoords in seperate thread 
                    print("refreshing buoy coords")
                    t = threading.Thread(target=self.BuoyCoordinates.getBuoyLocationsThreading, 
                                                args=(pred_dict["ship"][0], pred_dict["ship"][1],results_list, newBuoyCoords), daemon=True)
                    t.start()
                
                # extract relevant gt buoys for the current frame from the buoyCoords file
                filteredBuoys = self.getNearbyBuoys(pred_dict["ship"], buoyCoords)
                # filter buoy predictions (NNS & relative Thresholding)
                filteredPreds, idx_dict = self.filterPreds(pred_dict['ship'], pred_dict['buoy_predictions'], filteredBuoys)
                matching_results = []
                if len(filteredBuoys) > 0 and len(filteredPreds) > 0:
                    # pass extracted buoys and predictions to matching 
                    matching_results = self.matching(filteredPreds, filteredBuoys)

                # compute color based on matching results
                color_dict_preds = {}
                color_dict_gt = {}
                color_dict_id = {}
                for i, m in enumerate(matching_results):
                    # get ID of BB
                    idx_pred = idx_dict[m[1]]   # remap matching index to pred index
                    id = int(pred[idx_pred, 8])
                    if id not in color_assignment or id==-1:  # if id has no color so far add it to color dict
                        if id == -1:    # if box is not currently tracked (-1), assign unique negative id
                            id = -1*i
                        color_assignment = self.getColor(color_assignment, frame_id, id)
                        color = color_assignment[id]['color']
                    else:   # if id already has assigned color, use this color
                        color = color_assignment[id]['color']
                        color_assignment[id]['frame'] = frame_id
                    color_dict_preds[idx_pred] = color
                    color_dict_gt[m[0]] = color
                    color_dict_id[id] = color
                    self.computeMatchingConf(id, filteredBuoys[m[0]], color) # icrease conf of pred gt matched pair
                    self.distanceEstimator.drawBoundingBoxes(frame, pred[idx_pred].unsqueeze(0), color=color[:3], conf_thresh=self.conv_thresh)   # draw bounding boxes based on matched indices

                if rendering:
                    with lock:  # send data to rendering obj
                        self.RenderObj.setShipData(*pred_dict["ship"])
                        self.RenderObj.setPreds(pred_dict["buoy_predictions"], color_dict_preds)
                        self.RenderObj.setBuoyGT(filteredBuoys, color_dict_gt)

                self.decayMatchingConf()    # decay conf for all matched pairs

                # display FPS
                current_time = self.displayFPS(frame, current_time)

                self.prepareData(color_dict_id)
                if frame_id % 15 == 0:
                    self.updatePlots(fig, color_dict_id)
                    # lpThread = threading.Thread(target=lp.updatePlots, args=(), daemon=False)
                    # lpThread.start()

                # Display the frame (optional for real-time applications)
                cv2.imshow("Buoy Association", frame)
                frame_id += 1

            key = cv2.waitKey(1)
            # Press 'q' to exit the loop
            if key == ord('q'):
                break

            if key == 32:
                cv2.waitKey(-1)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


ba = BuoyAssociation()

# test_folder = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/Testdata/954_2_Pete2"
# images_dir = os.path.join(test_folder, 'images') 
# imu_dir = os.path.join(test_folder, 'imu') 
# ba.test(images_dir, imu_dir)

ba.video(video_path="/home/marten/Uni/Semester_4/src/TestData/954_2.avi", imu_path="/home/marten/Uni/Semester_4/src/TestData/furuno_954.txt", rendering=True)
#ba.video(video_path="/home/marten/Uni/Semester_4/src/TestData/videos_from_training/19_2.avi", imu_path="/home/marten/Uni/Semester_4/src/TestData/videos_from_training/furuno_19.txt", rendering=True)
#ba.video(video_path="/home/marten/Uni/Semester_4/src/TestData/22_2.avi", imu_path="/home/marten/Uni/Semester_4/src/TestData/furuno_22.txt", rendering=True)