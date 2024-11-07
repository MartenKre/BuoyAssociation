# Script to run Inference with Pytorch framework without relying on YOLOv7-dataloader

import os
import torch
import cv2
import numpy as np
import json
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, box_iou
from utils.datasets import letterbox
from utils.plots import plot_one_box


class DistanceEstimator():
    def __init__(self, weights = '/home/marten/Uni/Semester_4/src/BuoyAssociation/DistanceEstimator/weights/best.pt', img_size=1024, 
                 conv_thresh=0.4, iou_thresh=0.5):
        self.checkPath(weights)
        self.model = attempt_load('DistanceEstimator/weights/best.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.img_size = img_size
        self.conf_thresh = conv_thresh  # conf thresh for NMS
        self.iou_thresh = iou_thresh    # iou thresh for NMS

    def __call__(self, img):
        # runs inference of given image
        # image will be resized to self.img_size and normalized

        # pre processing
        img_resized, ratio, pad = letterbox(img, new_shape=(self.img_size, self.img_size), auto=False, scaleup=False, stride=32) # Resize with padding
        img_resized = img_resized[:,:,::-1].transpose(2, 0, 1).copy()  # BGR to RGB, shape to [3, height, width]
        img_resized = torch.from_numpy(img_resized).float() / 255.0
        img_resized = img_resized.unsqueeze(0).to('cuda')  # Add batch dimension

        # Run inference
        with torch.no_grad():
            pred, train_out = self.model(img_resized)  # Model predictions

        # post processing (NMS) & rescaling to original image
        detections = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)[0]
        detections[:, :4] = scale_coords(img_resized.shape[2:], detections[:, :4], img.shape, ratio_pad=(ratio, pad)).round()
        return detections.cpu()
    
    def checkPath(self, filepath):
        if not os.path.isfile(filepath):
            raise ValueError(f"Path to Model weights invalid: {filepath}")

    def plot_inference_results(self, pred, img, name, folder=None, labelsData = None, conf_thresh = 0.25):
        # Function to plot the BoundingBox from the Prediction vector onto the image
        # labelsData: list of Lables in YOLO-Format -> one list item for each BB label
        # individual BB label has format: [cls,x_center,y_center,w,h,dist] (all normalized except dist)
        if folder is None:
            folder = self.createDir(name='inference_results')
        if labelsData is not None:
            matchedPairs = self.findCorrespondance(labelsData, pred, img)
        for BB in pred:
            BB = BB.cpu()
            x,y,w,h,conf,cls,dist=BB
            if conf > conf_thresh: # only plot BBs that have conf values larger than 0.25
                if labelsData is None:
                    annotation = f"{conf:.2f}, {int(dist)}"
                else:
                    # check if BB is asscoiated with GT box
                    k = self.BB2HashKey(BB)
                    if k in matchedPairs:
                        targetDist = matchedPairs[k][-1]
                        annotation = f"{conf:.2f}, {int(dist)}/{int(targetDist)}"
                    else:
                        annotation = f"{conf:.2f}, {int(dist)}"
                plot_one_box([x,y,w,h], img, label=annotation)
        img_path = os.path.join(folder, name)
        cv2.imwrite(img_path, img)
        print(f"Image saved to: {img_path}")

    def createDir(self, name):
        script_path = os.path.abspath(__file__)
        path = os.path.join(os.path.dirname(script_path), name)
        os.makedirs(path, exist_ok=True)
        return path
    
    def BB2HashKey(self, BB):
        return "_".join([str(round(float(x), 3)) for x in BB])
    
    def findCorrespondance(self, labels, pred, img):
        # function gets all BB labels form labels file and BB prediction and returns the targetdist with the closest label
        image_height, image_width, image_channels = np.shape(img)
        absolute_gt_boxes = []
        for label in labels:
            cls, x_center, y_center, width, height, dist = label
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            absolute_gt_boxes.append([x1, y1, x2, y2, cls, dist])

        pred = pred.cpu()
        pred_boxes = pred[...,:4]
        gt_boxes_tensor = torch.tensor([box[:4] for box in absolute_gt_boxes])  # Only coordinates
        
        # Compute IoU between each predicted box and each ground truth box
        ious = box_iou(pred_boxes, gt_boxes_tensor)

        # Define IoU threshold for matching
        iou_threshold = 0.1
        matched_pairs = {}

        # Find matches based on IoU threshold
        for i, single_pred in enumerate(pred):
            best_iou, best_idx = torch.max(ious[i], 0)  # Find best IoU for each prediction
            if best_iou > iou_threshold:
                k = self.BB2HashKey(single_pred)
                matched_pairs[k] = absolute_gt_boxes[best_idx.item()]

        # Output matched pairs
        return matched_pairs
    
    def LabelsYOLOFormat(self, labelsFile):
        # function that constructs a labels list (float values) from txt file
        if os.path.exists(labelsFile):
            with open(labelsFile, 'r') as f:
                labels = f.readlines()
                for i, label in enumerate(labels):
                        label = label.split(" ")
                        if label[-1][-1:] == "\n":
                            label[-1] = label[-1][:-1]
                        labels[i] = [float(x) for x in label]
                return labels
        else:
            print("Label file {labelsFile} not fouond!")
            return None
        
    def LabelsJSONFormat(self, labelsFile):
        # function constructs labels list (including dist & buoylatlng) from JSON file
        result = []
        with open(labelsFile, 'r') as f:
            data = json.load(f)
            for i,obj in enumerate(data[0]["objects"]):
                cls = float(obj["type"])
                height = float(data[0]["height"])
                width = float(data[0]["width"])
                meta = data[1]["objects"][i]
                x_cetner = (float(meta["x1"]) + float(meta["x2"])) / 2 / width
                y_center = (float(meta["y1"]) + float(meta["y2"])) / 2 / height
                width_BB = (float(meta["x2"]) - float(meta["x1"])) / width
                height_BB = (float(meta["y2"]) - float(meta["y1"])) / height
                if "distanceGT" in meta["attributes"]:
                    dist = int(meta["attributes"]["distanceGT"])
                    latlng = [float(meta["attributes"]["buoyLat"]), float(meta["attributes"]["buoyLng"])]
                else:
                    print(f"DistanceGT missing in {labelsFile}")
                    dist = -1
                    latlng = None
                result.append([cls, x_cetner, y_center, width_BB, height_BB, dist, latlng])
        return result

