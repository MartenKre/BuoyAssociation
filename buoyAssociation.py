# Script uses DistanceEstimator to obtain Object Detection DistEst Data from modified YOLOv7 Network
  
import cv2
import sys
import os
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DistanceEstimator'))

from DistanceEstimator import DistanceEstimator

test_folder = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/StPete_BuoysOnly/956_2/"
images_dir = os.path.join(test_folder, 'images') 
labels_dir = os.path.join(test_folder, 'labels')
imu_dir = os.path.join(test_folder, 'imu') 

class BuoyAssociation():
    def __init__(self, focal_length=0.275, pixel_size=1.55e-3, img_sz=[1920, 1080]):
        self.focal_length = focal_length        # focal length of camera in mm
        self.scale_factor = 1 / (2*pixel_size)  # scale factor of camera -> pixel size in mm
        self.image_size = img_sz
        self.distanceEstimator = DistanceEstimator(iou_thresh = 0.3)
        self.imu_data = self.getIMUData()
        self.runAssociations()

    def runAssociations(self):
        for image in os.listdir(images_dir):
            self.getPredictions(os.path.join(images_dir, image))

    def getPredictions(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read {image_path}")
            return False
        pred = self.distanceEstimator(img)
        
        # get pixel center coordinates of BBs and compute lateral angle
        pred = self.getAnglesOfIncidence(pred)
        # compute buoy predictions
        # self.BuoyLocationPred(pred, )

        labels = self.getLabelsData(image_path)
        labels_wo_gps = [x[:-1] for x in labels]
        #self.distanceEstimator.plot_inference_results(pred, img, name=os.path.basename(image_path), 
        #                                              folder="/home/marten/Uni/Semester_4/src/BuoyAssociation/detections", labelsData=labels_wo_gps)

    def getAnglesOfIncidence(self, preds):
        # function returns angle (radians) of deviation between optical axis of camera and object in x (horizontal) direction
        # concatenates prediction tensor with angle at last index -> tensor has shape [n, 8]
        x_center = preds[:,0]
        u_0 = self.image_size[0] / 2
        x = (x_center - u_0) / self.scale_factor
        alpha = torch.arctan((x)/self.focal_length).unsqueeze(1)
        preds = torch.cat((preds, alpha), dim=-1)
        return preds
    
    def BuoyLocationPred(self, frame_id, preds):
        # for each BB prediction function computes the Buoy Location based on Dist & Angle of tensor
        latCam = self.imu_data[frame_id][3]
        lngCam = self.imu_data[frame_id][4]
        #preds[]

    def getLabelsData(self, image_path):
        labelspath = os.path.join(labels_dir, os.path.basename(image_path) + ".json")
        if os.path.exists(labelspath):
            return self.distanceEstimator.LabelsJSONFormat(labelspath)
        else:
            print(f"LablesFile not found: {labelspath}")
            return None
        
    def getIMUData(self):
        # functino returns IMU data as list
        files = os.listdir(imu_dir)
        filename = [f for f in files if f.endswith('.txt')][0]
        path = os.path.join(imu_dir, filename)
        result = []
        with open(path, 'r') as f:
            data = f.readlines()
            for line in data:
                line = [float(x) for x in line]
                result.append(line)
        if len(result) == 0:
            print("No IMU data found, check path: {path}")
        return result

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


ba = BuoyAssociation()
