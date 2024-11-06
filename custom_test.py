# Script to run Inference with Pytorch framework without relying on YOLOv7-dataloader

import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DistanceEstimator'))

from DistanceEstimator import DistanceEstimator

test_images_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/images"
test_labels_dir = "/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/test/labels"
distEst = DistanceEstimator()

for image_name in os.listdir(test_images_dir):
    # Load image
    image_path = os.path.join(test_images_dir, image_name)
    img = cv2.imread(image_path)
    with open(os.path.join(test_labels_dir, image_name.replace("png", "txt")), 'r') as f:
        labels = f.readlines()
        if img is None:
            print(f"Could not read {image_path}")
            continue
        pred = distEst(img)
        distEst.plot_inference_results(pred, img, image_name, labelsData=labels)