import json
from os.path import isfile
import cv2
import sys
import os
import numpy as np
import torch
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment, minimize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torchvision.ops.boxes import box_area
from pprint import pprint
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DistanceEstimator'))
from DistanceEstimator import DistanceEstimator

from utility.Transformations import ECEF2LatLng, T_ECEF_Ship, LatLng2ECEF, haversineDist
from utility.GeoData import GetGeoData
from boxmot import ByteTrack

class BuoyAssociation():    
    def __init__(self, focal_length=2.75, pixel_size=0.00155, img_sz=[1920, 1080]):
        self.focal_length = focal_length        # focal length of camera in mm
        self.scale_factor = 1 / (2*pixel_size)  # scale factor of camera -> pixel size in mm
        self.image_size = img_sz
        self.conf_thresh = 0.25  # used for NMS and BoxMOT (x2) and drawing -> Detections below this thresh won't be considered
        self.distanceEstimator = DistanceEstimator(img_size = 1024, conv_thresh = self.conf_thresh, iou_thresh = 0.2)    # load Yolov7 with Distance Module, conv & iou_thresh for NMS
        self.BuoyCoordinates = GetGeoData(tile_size=0.02) # load BuoyData from GeoJson
        self.track_buffer = 60   # after exceeding thresh (frames count) a lost will be reassigned new ID 
        self.MOT = self.initBoxMOT()        # Multi Object Tracker Instance
        self.ma_storage = {}    # dict to store moving averages
        self.matching_confidence = {}
        self.matching_confidence_plotting = {}
        self.fl_bias = [0]  # focal length bias
        self.heading_bias = [0] # heading bias
        self.use_biases = True

        # Hyperparameters Filtering Pre/ Port Matching
        self.dist_thresh_far = 0.2
        self.dist_thresh_close = 0.35
        self.isclose = 150

        self.metrics = {"tp": 0, "fp": 0, "fn": 0}


    def test(self, test_dir, video=False):
        # load IMU data
        imu_file = os.path.join(test_dir, "imu_data.json")
        with open(imu_file, "r") as f:
            imu_data = json.load(f)

        images_dir = os.path.join(test_dir, "images")
        labels_dir = os.path.join(test_dir, "labels")

        if not video:
            self.use_biases = False


        frame_id = 0
        for image in tqdm(os.listdir(images_dir)):   
            image_path = os.path.join(images_dir, image) 
            img = cv2.imread(image_path)
            ship_pose = imu_data[image.replace(".png", "")]

            # get inference predictions
            if video:
                pred, pred_dict = self.getPredictions(img, frame_id, ship_pose, moving_average=True, boxMOT=True)
            else:
                pred, pred_dict = self.getPredictions(img, frame_id, ship_pose, moving_average=False, boxMOT=False)

            # get nearby buoys 
            buoyCoords = self.BuoyCoordinates.getBuoyLocations(ship_pose[0], ship_pose[1])

            # extract relevant gt buoys for the current frame from the buoyCoords file
            filteredBuoys = self.getNearbyBuoys(ship_pose, buoyCoords)
            # filter buoy predictions (NNS & relative Thresholding)
            filteredPreds, idx_dict = self.filterPreds(pred_dict['ship'], pred_dict['buoy_predictions'], filteredBuoys)
            matching_results = []
            if len(filteredBuoys) > 0 and len(filteredPreds) > 0:
                # pass extracted buoys and predictions to matching 
                matching_results = self.matching(filteredPreds, filteredBuoys)
                # different buoy far away -> remove this bb from the matching results
                matching_results = self.postprocessMatching(matching_results, filteredBuoys, filteredPreds, pred_dict['ship'])
            
            # go through matched buoys
            pred_results = []
            matched_pairs = {}  # dict that contains the matched pairs (pred / buoy gt) -> lat/lng coords and pred idx
            for i, m in enumerate(matching_results):
                # get ID of pred
                idx_pred = idx_dict[m[1]]
                idx_buoyGT = m[0]

                if video:
                    self.computeMatchingConf(int(pred[idx_pred, 8]), filteredBuoys[m[0]]) # icrease conf of pred gt matched pair

                bb_pred = pred[idx_pred, 0:4]   # get BB coordinates
                id = self.BuoyCoordinates.getBuoyID(*filteredBuoys[idx_buoyGT])  # get ID of matchted GT buoy

                pred_results.append(torch.cat((bb_pred, torch.tensor([id]))))
                matched_pairs[int(pred[idx_pred, 8])] = (filteredPreds[m[1]], filteredBuoys[m[0]], idx_pred)
            if len(pred_results) > 0:
                pred_results = torch.stack(pred_results)
                pred_results = self.xyxy2cxcywh(pred_results, img_dims=self.image_size)
            else:
                pred_results = torch.zeros((0, 5))

            if video:
                self.decayMatchingConf()    # decay conf for all matched pairs

            if video:
                # compute heading bias & focal length delta
                self.correctCameraBias(matched_pairs, pred_dict['ship'], pred)

            # get corresponding labels file 
            labels = self.loadLabels(os.path.join(labels_dir, image.replace(".png", ".txt")))
            self.computeMetrics(labels, pred_results)

            frame_id += 1

        print("Testing Done")
        print("Results:")
        pprint(self.metrics)


    def computeMetrics(self, labels, preds, iou_thresh=0.5):
        """ Computes Metrics (TP,FP,FN) for given preds and labels containing normalized BB coords and BuoyIDs
        Args:
            labels: tensor (n, 5) -> [cx, cy, w, h, buoy_id]
            preds: tensor (m, 5) -> [cx, cy, w, h, buoy_id]
            iou_thresh: threshold to consider an object as detected based on intersection over union 
        """
        tp = 0
        fp = 0
        fn = 0
        for label in labels:  # check labels
            matched = False
            for pred in preds:
                if label[-1] == pred[-1]:   # if id in pred and label match, check for iou
                    bb_l = self.box_cxcywh_to_xyxy(label[:-1].unsqueeze(0))
                    bb_p = self.box_cxcywh_to_xyxy(pred[:-1].unsqueeze(0))
                    if self.box_iou(bb_l, bb_p) > iou_thresh:
                        tp += 1     # if iou_thresh is exceeded, pred is tp
                    else:           # if iou_thresh is not exceeded
                        fn += 1     # label is fn, since no correct matched pred exists
                        fp += 1     # pred with bouy id is fp since in wrong place
                    matched=True
                    break
            if not matched:
                fn += 1

        covered_ids = labels[:,-1]
        for pred in preds:  # check remaining preds -> predictions that do not occur in labels are FP
            if pred[-1] not in covered_ids:
                fp += 1

        self.metrics["tp"] += tp
        self.metrics["fp"] += fp
        self.metrics["fn"] += fn
       
        
    def box_iou(self, boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou

    def xyxy2cxcywh(self, x, img_dims=[1920, 1080]):
        w, h = img_dims
        x0, y0, x1, y1, id = x.unbind(-1)
        b = [(x0 + x1) / (2*w), (y0 + y1) / (2*h),
            (x1 - x0) / w, (y1 - y0) / h, id]
        return torch.stack(b, dim=-1)

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)

        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def loadLabels(self, path_to_file):
        if not isfile(path_to_file):
            raise ValueError(f"Labels File {path_to_file} does not exits")

        data = []
        with open(path_to_file, "r") as f:
            data = f.read().splitlines()

        res = torch.zeros((len(data), 5))
        for i,line in enumerate(data):
            elems = line.split(" ")
            elems = [float(x) for x in elems]
            res[i,:] = torch.tensor(elems, dtype=torch.float32)

        return res


    def getPredictions(self, img, frame_id, ship_pose, moving_average=True, boxMOT=True):
        """Function runs inferense, returns Yolov7 preds concatenated with bearing to Objects & ID of BoxMOT
        Prediction dict contains ship pose and predicted buoy positions (in lat,lng)
        Args:
            Img: pixel array
            frame_id: current frame ID
            ship_pose: list [lat, lng, heading]
        Returns:
            preds: (n,8) tensor [xyxy, conf, cls, dist, angle, ID]
            pred_dict dict containing "ship": [lat, lng, heading] and "buoy_predictions": [[lat1,lng1], ...]
        """

        pred = self.distanceEstimator(img)
        pred = self.getAnglesOfIncidence(pred)  # get pixel center coordinates of BBs and compute lateral angle
        
        # BoxMOT tracking on predictions
        if boxMOT:
            preds_boxmot_format = pred[:,0:6]    # preds need to be in format [xyxy, conf, classID]
            preds_boxmot_format = np.array(preds_boxmot_format)
            res = self.MOT.update(preds_boxmot_format, img)   # res --> M X (x, y, x, y, id, conf, cls, ind)
            res = torch.from_numpy(res)
            ids = -1*torch.ones(size=(pred.size()[0], 1))   # default case -1
            if len(res) > 0:
                ids[res[:,-1].to(torch.int32), 0] = res[:, 4].to(torch.float)
            pred = torch.cat((pred, ids), dim=-1)   # pred = [N x [xyxy,conf,classID,dist,angle,ID]]
        else:
            ids = -1*torch.ones(size=(pred.size()[0], 1))   # default case -1
            pred = torch.cat((pred, ids), dim=-1)   # pred = [N x [xyxy,conf,classID,dist,angle,ID]]

        if moving_average:
            pred = self.moving_average(pred, frame_id, method='EMA')

        pred_dict = self.buoyLocationPred(pred, ship_pose)   # compute buoy predictions
        return pred, pred_dict
        
    def getAnglesOfIncidence(self, preds):
        """Computes angle of deviation between optical axis of cam and object in x (horizontal direction)

        Returns:
            prediction tensor concatenated with angle: on (n,8) tensor per image [xyxy, conf, cls, dist, angle]
        """

        x = self.pixel2mm(preds)
        if self.use_biases: # use computed focal length bias for computation of bearing
            fl_bias = self.computeEma(self.fl_bias)
            alpha = -1*torch.arctan((x)/(self.focal_length+fl_bias)).unsqueeze(1)
        else:
            alpha = -1*torch.arctan((x)/self.focal_length).unsqueeze(1)
        preds = torch.cat((preds, alpha), dim=-1)
        return preds   
    
    def pixel2mm(self, preds):
        # function converts pixel coordinates of bounding boxes to mm

        x_center = (preds[:,0] + preds[:,2]) / 2    # convert left and right corner x coords to center coord x
        u_0 = self.image_size[0] / 2
        x = (x_center - u_0) / self.scale_factor
        return x
    
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

    def computeEma(self, sequence, alpha = 0.9):
        # function computes ema on a sequence of values

        i = len(sequence)
        result = alpha**i * sequence[0]
        for j in range(0,i):
            result += sequence[i-1-j] * (1-alpha) * alpha ** j
        return result
    
    def buoyLocationPred(self, preds, ship_pose):
        """for each BB prediction function computes the Buoy Location based on Dist & Angle of the tensor
        Args:
            preds: prediction tensor of yolov7 (N,8) -> [Nx[xyxy, conf, cls, dist, angle]]
            ship_pose: [lat, lng, heading]
        Returns:
            Dict{"ship:"[lat,lng,heading], "buoy_prediction":[[lat1,lng1],[lat2,lng2]]}
        """

        latCam = ship_pose[0]
        lngCam = ship_pose[1]
        heading = ship_pose[2]
        if self.use_biases:
            heading_bias = self.computeEma(self.heading_bias)
            heading = heading - np.rad2deg(heading_bias)

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

    def initBoxMOT(self):
        return ByteTrack(
            track_thresh=2 * self.conf_thresh,      # threshold for detection confidence -> seperates BBs into high and low confidence
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
    
    def Coords2Hash(self, coords):
        # takes a tuple of lat, lng coordinates and returns a unique hash key as string
        return str(coords[0])+str(coords[1])
    
    def computeMatchingConf(self, id, buoyGT): 
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

    def correctCameraBias(self, matched_pairs, ship, preds):
        """Computes heading and Focal Length bias based on bearing of pred & GT buoy pairs
        Args:
            matched_pairs: List containing [buoyPred(Lat,Lng), buoyGT(Lat,Lng), idx_pred_tensor]
            ship: lat, lng heading
            preds: prediction tensor
        """
        self.computeFLBias(matched_pairs, ship, preds)
        self.computeHeadingBias(matched_pairs, ship, preds)

    def computeFLBias(self, matched_pairs, ship, preds):
        # Function computes Focal Length bias

        # first check if all criteria are met to correct Bias
        # Criteria:
        #   1) At least two buoys in view (set)
        #   2) Set of buoys must consist of pos & neg bearing
        #   3) Distance to buoys should not be greater than certain threshold
        #   4) Dist Prediction error is smaller than certain threshold
        #   5) Confidence of matching must exceed certain threshold
        dist_abs_thresh = 300
        match_conf_thresh = 4   # 1/15 per frame -> matched time > 4 sec
        dist_err_thresh = 0.2   # relative error distance pred to distance gt

        filtered_pairs = []
        for key in matched_pairs:
            dist_gt = haversineDist(*matched_pairs[key][1], *ship[0:2])
            dist_pred = haversineDist(*matched_pairs[key][0], *ship[0:2])
            k = self.Coords2Hash(matched_pairs[key][1])
            if k not in self.matching_confidence or key not in self.matching_confidence[k]: 
                continue    # if k (Hash Key of buoy GT) not in matching_confidence -> skip
            if ((self.matching_confidence[k][key] >= match_conf_thresh) and (dist_gt < dist_abs_thresh) 
                and ((abs(dist_pred - dist_gt) / dist_gt) < dist_err_thresh)):
                filtered_pairs.append(matched_pairs[key])

        # check that bearing to left / right of boat heading exists
        x,y,z = LatLng2ECEF(*ship[0:2])
        SHIP_T_ECEF = np.linalg.pinv(T_ECEF_Ship(x,y,z,ship[2]))
        bearings = []
        for pred, gt, idx in filtered_pairs:
            x,y,z = LatLng2ECEF(*pred)
            P_Pred_Ship = SHIP_T_ECEF@np.array([x,y,z,1])
            bearing_pred = np.arctan2(P_Pred_Ship[1], P_Pred_Ship[0])
            x,y,z = LatLng2ECEF(*gt)
            P_GT_Ship = SHIP_T_ECEF@np.array([x,y,z,1])
            bearing_gt = np.arctan2(P_GT_Ship[1], P_GT_Ship[0])
            # make sure both bearings have same sign, since this leads to problems with gradient descent
            # e.g. bearing_gt = -0.1, but bearing_pred = 0.1 -> since we are only allowed to change focal length
            # gradient descent would set focal_length to infinity, thus reducing bearing_pred to zero (optimum)
            if np.sign(bearing_gt) == np.sign(bearing_pred):
                bearings.append([bearing_pred, bearing_gt, idx])

        bearings = np.asarray(bearings)
        if bearings.shape[0] == 0 or np.min(bearings[:, 0]) > 0 or np.max(bearings[:,0]) < 0:
            return  # if bearings are only left or only right of principal ray -> exit

        def errorFunctionFL(params, x, theta, focal_length):  # error function for optimizer
            delta_f= params
            error = (-1 * np.arctan(x / (focal_length + delta_f)) - theta)**2
            error = np.sum(error)
            return error
        
        delta_f = self.fl_bias[-1] # delta focal length (initial guess, starts with zero if no optimizations were successful so far)
        x = self.pixel2mm(preds[bearings[:,2],:]).numpy()  # bb center_x in mm
        theta = bearings[:,1]   # target angle (buoy gt)

        result = minimize(errorFunctionFL, (delta_f), args=(x, theta, self.focal_length))    # gradient descent
        delta_f = result.x[0]
        if len(self.fl_bias) == 1 and self.fl_bias[0] == 0:
            self.fl_bias[0] = delta_f   # if first bias value -> overwrite default
        else:
            self.fl_bias.append(delta_f)    # else append

    def computeHeadingBias(self, matched_pairs, ship, preds):
        # function computes heading bias

        # first check if all criteria are met to correct Bias
        # Criteria:
        #   1) At least one buoy direcly in center (only x coord) of camera view 
        #   2) Distance to buoy should not be greater than certain threshold
        #   3) Dist Prediction error is smaller than certain threshold
        #   4) Confidence of matching must exceed certain threshold
        dist_abs_thresh = 300
        match_conf_thresh = 4   # 1/15 per frame -> matched time > 4 sec
        dist_err_thresh = 0.2   # relative error distance pred to distance gt
        center_thresh = 2.5     # a buoy is in center if its bearing does not exceed 5 degrees

        filtered_pairs = []
        for key in matched_pairs:   # filter matched pairs based on distance err, abs dist and confidence in matching
            dist_gt = haversineDist(*matched_pairs[key][1], *ship[0:2])
            dist_pred = haversineDist(*matched_pairs[key][0], *ship[0:2])
            k = self.Coords2Hash(matched_pairs[key][1])
            if k not in self.matching_confidence or key not in self.matching_confidence[k]: 
                continue    # if k (Hash Key of buoy GT) not in matching_confidence -> skip
            if ((self.matching_confidence[k][key] >= match_conf_thresh) and (dist_gt < dist_abs_thresh) 
                and ((abs(dist_pred - dist_gt) / dist_gt) < dist_err_thresh)):
                filtered_pairs.append(matched_pairs[key])

        # only filter pairs that are directly in front of camera (i.e. bearing does not exceed center_thresh)
        x,y,z = LatLng2ECEF(*ship[0:2])
        SHIP_T_ECEF = np.linalg.pinv(T_ECEF_Ship(x,y,z,ship[2]))
        bearings = []
        for pred, gt, idx in filtered_pairs:
            x,y,z = LatLng2ECEF(*pred)
            P_Pred_Ship = SHIP_T_ECEF@np.array([x,y,z,1])
            bearing_pred = np.arctan(P_Pred_Ship[1] / P_Pred_Ship[0])
            x,y,z = LatLng2ECEF(*gt)
            P_GT_Ship = SHIP_T_ECEF@np.array([x,y,z,1])
            bearing_gt = np.arctan(P_GT_Ship[1] / P_GT_Ship[0])

            if abs(np.rad2deg(bearing_pred)) <= center_thresh:
                bearings.append([bearing_pred, bearing_gt, idx])

        bearings = np.asarray(bearings)
        if bearings.shape[0] == 0: 
            return  # if no samples left return

        def errorFunctionHeading(params, alpha, theta):
            delta_h = params
            error = (theta - alpha - delta_h)**2
            error = np.sum(error)
            return error

        delta_h = self.heading_bias[-1] # delta heading (initial guess)
        alpha = bearings[:, 0]
        theta = bearings[:, 1]
        result = minimize(errorFunctionHeading, (delta_h), args=(alpha, theta)) # gradient descent
        delta_h = result.x[0]
        if len(self.heading_bias) == 1 and self.heading_bias[0] == 0:
            self.heading_bias[0] = delta_h   # if first bias value -> overwrite default
        else:
            self.heading_bias.append(delta_h)    # else append
            
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

    def getNearbyBuoys(self, ship_pose, buoyCoords, fov_with_padding=120, dist_thresh=1200, nearby_thresh=50):
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
            bearing = np.rad2deg(np.arctan2(pos_bouy[1],pos_bouy[0]))   # compute bearing of buoy
            dist_to_ship = haversineDist(lat, lng, ship_pose[0], ship_pose[1])  # compute dist to ship
            if abs(bearing) <= fov_with_padding / 2 and dist_to_ship <= dist_thresh:
                # include buoys that are within fov+padding and inside maxdist
                selected_buoys.append((lat, lng))
            elif dist_to_ship <= nearby_thresh:
                # also include nearby buoys not inside FOV
                selected_buoys.append((lat, lng))

        return list(set(selected_buoys))

    def filterPreds(self, shipPose, preds, buoysGT):
        """ function filters predictions based on nearest neighbour search & thresholding
        thrshold is based on relative distance from pred to ship and distinguished between far (>150m) and close (<=150m)
        Args:
            shipPose:       [lat,lng,heading]
            preds:          list of buoy preds in lat lng format [lat,lng],...]
            buoysGT:        list of ground truth buoys in lat lng format [[lat,lng],...]
        Returns:
            filtered_preds: filtered list of buoy predictions in lat lng format [[lat,lng],...]
            index_dict: matches original indices of prediction vector to new indices of filtered preds 
        """

        filtered_preds = []
        index_dict = {}
        if buoysGT == []:
            return filtered_preds, index_dict
        for i, pred in enumerate(preds):
            distances = list(map(lambda x: haversineDist(*pred, *x), buoysGT))
            closestGT = np.argmin(distances)
            closest_dist = min(distances) # closest dist to gt buoys
            dist_to_ship = haversineDist(*pred, *shipPose[:2])  # dist between prediciton and ship
            dist_gt_ship = haversineDist(*buoysGT[closestGT], *shipPose[:2])    # dist between gt buoy pos and ship
            dist_thresh = self.dist_thresh_far if dist_gt_ship > self.isclose else self.dist_thresh_close
            if closest_dist / dist_to_ship <= dist_thresh:
                filtered_preds.append(pred)
                index_dict[len(filtered_preds)-1] = i
        return filtered_preds, index_dict
    
    def postprocessMatching(self, matchingResults, filteredBuoys, filteredPreds, ship):
        # filters matching results based on distance between matched pred & gt pair
        # if dist exceeds dist thresh (absolute value) then not valid matching

        matchings_filtered = []
        for idx_gt, idx_pred in matchingResults:
            buoyGT = filteredBuoys[idx_gt]
            buoyPred = filteredPreds[idx_pred]

            dist_buoys = haversineDist(*buoyGT, *buoyPred)
            dist_gt_ship = haversineDist(*buoyGT, *ship[0:2])
            dist_thresh = self.dist_thresh_far if dist_gt_ship > self.isclose else self.dist_thresh_close
            if dist_buoys / dist_gt_ship <= dist_thresh:
                matchings_filtered.append((idx_gt, idx_pred))

        return matchings_filtered


ba = BuoyAssociation()

ba.test(test_dir="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/YOLO_Testset")
