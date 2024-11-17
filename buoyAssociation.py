# Script uses DistanceEstimator to obtain Object Detection DistEst Data from modified YOLOv7 Network
import cv2
import sys
import os
import numpy as np
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import threading
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utility.Transformations import ECEF2LatLng, T_ECEF_Ship, LatLng2ECEF, haversineDist
from utility.GeoData import GetGeoData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DistanceEstimator'))

from DistanceEstimator import DistanceEstimator

test_folder = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/Testdata/954_2_Pete2"
images_dir = os.path.join(test_folder, 'images') 
labels_dir = os.path.join(test_folder, 'labels')
imu_dir = os.path.join(test_folder, 'imu') 

class BuoyAssociation():
    def __init__(self, focal_length=2.75, pixel_size=0.00155, img_sz=[1920, 1080]):
        self.focal_length = focal_length        # focal length of camera in mm
        self.scale_factor = 1 / (2*pixel_size)  # scale factor of camera -> pixel size in mm
        self.image_size = img_sz
        self.distanceEstimator = DistanceEstimator(conv_thresh = 0.4, iou_thresh = 0.3)    # load Yolov7 with Distance Module, iou_thresh for NMS
        self.BuoyCoordinates = GetGeoData(tile_size=0.02) # load BuoyData from GeoJson
        self.imu_data = None 

    def test(self, plots=True):
        # function tests performance of BuoyAssociation on Labeled Set of Images including BuoyGT
        # plots will include distance GT and GT buoy positions
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
            pred, pred_dict = self.getPredictions(img, idx)

            labels = self.getLabelsData(image_path)
            labels_wo_gps = [x[:-1] for x in labels]
            buoyLabels = [x[-1] for x in labels]
            pred_dict["buoys_labeled"] = buoyLabels      

            if plots:
                self.distanceEstimator.plot_inference_results(pred, img, name=os.path.basename(image_path), 
                                                                folder=plots_folder, labelsData=labels_wo_gps)
                self.plot_Predictions(pred_dict, name = os.path.basename(image_path).replace(".png", ""), 
                                            folder=plots_folder)

    def getPredictions(self, img, frame_id):
        # Arguments: Img as Pixel array and current frame_id 
        # Function runs inference, returns Yolov7 preds concatenated with bearing 
        # and prediction dict containing ship location,heading and bouy preds in lat, lng
        pred = self.distanceEstimator(img)
        # get pixel center coordinates of BBs and compute lateral angle
        pred = self.getAnglesOfIncidence(pred)
        # compute buoy predictions
        pred_dict = self.BuoyLocationPred(frame_id, pred)
        return pred, pred_dict
        
    def getAnglesOfIncidence(self, preds):
        # function returns angle (radians) of deviation between optical axis of camera and object in x (horizontal) direction
        # concatenates prediction tensor with angle at last index -> tensor has shape [n, 8]
        # objects to the left of the optical axis have positive angle, objects to the right have a negative one
        x_center = preds[:,0]
        u_0 = self.image_size[0] / 2
        x = (x_center - u_0) / self.scale_factor
        alpha = -1*torch.arctan((x)/self.focal_length).unsqueeze(1)
        preds = torch.cat((preds, alpha), dim=-1)
        return preds
    
    def BuoyLocationPred(self, frame_id, preds):
        # for each BB prediction function computes the Buoy Location based on Dist & Angle of the tensor
        latCam = self.imu_data[frame_id][3]
        lngCam = self.imu_data[frame_id][4]
        heading = self.imu_data[frame_id][2]

        # trasformation:    latlng to ecef, ecef to enu, enu to ship
        x, y, z = LatLng2ECEF(latCam, lngCam)  # ship coords in ECEF
        ECEF_T_Ship = T_ECEF_Ship(x,y,z,heading)   # transformation matrix between ship and ecef

        # compute 2d points (x,y) in ship cs, (z=0, since all objects are on water surface)
        buoysX = (torch.cos(preds[:,-1]) * preds[:,-2]).tolist()
        buoysY = (torch.sin(preds[:,-1]) * preds[:,-2]).tolist()
        buoy_preds = list(zip(buoysX, buoysY))

        # transform buoyCoords to lat lng
        buoysLatLng = []
        for buoy in buoy_preds:
            p = ECEF_T_Ship @ np.array([buoy[0], buoy[1], 0, 1])    # buoy coords in ecef
            lat, lng, alt = ECEF2LatLng(p[0],p[1],p[2])
            buoysLatLng.append((lat, lng))

        return {"buoy_predictions": buoysLatLng, "ship": [latCam, lngCam, heading]}

    def getLabelsData(self, image_path):
        labelspath = os.path.join(labels_dir, os.path.basename(image_path) + ".json")
        if os.path.exists(labelspath):
            return self.distanceEstimator.LabelsJSONFormat(labelspath)
        else:
            print(f"LablesFile not found: {labelspath}")
            return None
        
    def getIMUData(self, path):
        # functino returns IMU data as list
        if os.path.isfile(path):
            result = []
            with open(path, 'r') as f:
                data = f.readlines()
                for line in data:
                    content = line.split(",")
                    line = [float(x) for x in content]
                    result.append(line)
        else:
            files = os.listdir(imu_dir)
            filename = [f for f in files if f.endswith('.txt')][0]
            path = os.path.join(imu_dir, filename)
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
    
    def inference_with_test_data(self):
        distEst = DistanceEstimator()
        test_images_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/images"
        test_labels_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/labels"
        for image_name in os.listdir(test_images_dir):
            # Load image
            image_path = os.path.join(test_images_dir, image_name)
            img = cv2.imread(image_path)
            labels = distEst.LabelsYOLOFormat(os.path.join(test_labels_dir, image_name.replace(".png", ".txt")))
            if img is None:
                print(f"Could not read {image_path}")
                continue
            pred = distEst(img)
            distEst.plot_inference_results(pred, img, image_name, labelsData=labels)

    def create_run_directory(self, base_name="run", path=""):
        i = 0
        while True:
            # Construct the folder name: 'run', 'run1', 'run2', etc.
            folder_name = f"{base_name}{i if i > 0 else ''}"
            
            # Check if the folder exists
            if not os.path.exists(os.path.join(path, folder_name)):
                path_to_folder = os.path.join(path, folder_name)
                # Create the folder if it doesn't exist
                os.makedirs(path_to_folder)
                print(f"Created directory: {path_to_folder} to store plots")
                return path_to_folder
            
            # Increment the suffix if the folder exists
            i += 1

    def matching(self, preds, buoys_chart):
        # function computes bipartite matching between chart buoys and predictions
        # Arguments: lists of the form: [[lat,lon], [lat,lon], ...] for preds and buoys_chart
        G = []  # cost matrix
        for pred in preds:
            edges = list(map(lambda x: int(haversineDist(*x, *pred)), buoys_chart))
            G.append(edges)

        G = np.asarray(G)
        row_ind, col_ind = linear_sum_assignment(G)     # run hungarian algorithm
        result = [(a,b) for a,b in zip(row_ind, col_ind)]   # create tuples of matched indices
        return result

    def getNearbyBuoys(self, ship_pose, buoyCoords, fov_with_padding=100, dist_thresh=1200, nearby_thresh=50):
        # function selects nearby buoys (relative to ship pos) from a list containing buoy Coordinates
        # fov with padding specifies horizontal fov of cam (95) with additional padding of 2.5 on both sides 
        # dist_thresh: if bouy exceeds this, it will not be included
        # nearby_thresh: even if bouy is not in fov it will still be included if inside this thresh -> chart data inaccuracies
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
                selected_buoys.append([lat, lng])
            elif dist_to_ship <= nearby_thresh:
                # also include nearby buoys not inside FOV
                selected_buoys.append([lat, lng])

        return selected_buoys

    def processVideo(self, video_path, imu_path):
        # load IMU data
        self.imu_data = self.getIMUData(imu_path)

        # load geodata
        lat_start = self.imu_data[0][3]
        lng_start = self.imu_data[0][4]
        buoyCoords = self.BuoyCoordinates.getBuoyLocations(lat_start, lng_start)
        self.BuoyCoordinates.plotBuoyLocations(buoyCoords)

        cap = cv2.VideoCapture(video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # get predictions for frame
            pred, pred_dict = self.getPredictions(frame, frame_id)
            # draw BBs on frame
            self.distanceEstimator.drawBoundingBoxes(frame, pred)

            # check if buoydata needs to be reloaded
            refresh = self.BuoyCoordinates.checkForRefresh(pred_dict["ship"][0], pred_dict["ship"][1])
            if refresh:
                # load new buoycoords in seperate thread 
                print("refreshing buoy coords")
                buoyCoords = threading.Thread(target=self.BuoyCoordinates.getBuoyLocations, 
                                              args=(pred_dict["ship"][0], pred_dict["ship"][1],), daemon=True)
            
            # extract relevant buoys for the current frame from the buoyCoords file
            filteredBuoys = self.getNearbyBuoys(pred_dict["ship"], buoyCoords)
            if len(filteredBuoys) > 0:
                # pass extracted buoys and predictions to matching 
                matching_results = self.matching(pred_dict["buoy_predictions"], filteredBuoys)
                
                # display results by plotting or live rendering

            # Display the frame (optional for real-time applications)
            cv2.imshow("Frame", frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

ba = BuoyAssociation()
#ba.test()
ba.processVideo(video_path="/home/marten/Uni/Semester_4/src/TestData/954_2.avi", imu_path="/home/marten/Uni/Semester_4/src/TestData/furuno_954.txt")
#ba.testmatching()