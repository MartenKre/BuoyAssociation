# Script uses DistanceEstimator to obtain Object Detection DistEst Data from modified YOLOv7 Network
  
import cv2
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DistanceEstimator'))

from DistanceEstimator import DistanceEstimator

test_folder = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/StPete_BuoysOnly/956_2/"
images_dir = os.path.join(test_folder, 'images') 
labels_dir = os.path.join(test_folder, 'labels')
imu_dir = os.path.join(test_folder, 'imu') 

class BuoyAssociation():
    def __init__(self):
        self.distanceEstimator = DistanceEstimator(iou_thresh = 0.3)
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
        labels = self.getLabelsData(image_path)
        labels_wo_gps = [x[:-1] for x in labels]
        self.distanceEstimator.plot_inference_results(pred, img, name=os.path.basename(image_path), 
                                                      folder="/home/marten/Uni/Semester_4/src/BuoyAssociation/detections", labelsData=labels_wo_gps)

    def getLabelsData(self, image_path):
        labelspath = os.path.join(labels_dir, os.path.basename(image_path) + ".json")
        if os.path.exists(labelspath):
            return self.distanceEstimator.LabelsJSONFormat(labelspath)
        else:
            print(f"LablesFile not found: {labelspath}")
            return None

    def inference_with_test_data(self):
        distEst = DistanceEstimator()
        test_images_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/images"
        test_labels_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/labels"
        for image_name in os.listdir(test_images_dir):
            # Load image
            image_path = os.path.join(test_images_dir, image_name)
            img = cv2.imread(image_path)
            labels = distEst.LabelsYOLOFormat(os.path.join(test_labels_dir, image_name.replace("png", "txt")))
            if img is None:
                print(f"Could not read {image_path}")
                continue
            pred = distEst(img)
            distEst.plot_inference_results(pred, img, image_name, labelsData=labels)


ba = BuoyAssociation()
