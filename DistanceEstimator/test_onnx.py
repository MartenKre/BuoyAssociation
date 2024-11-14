# Inference for ONNX model without Yolov7 dataloader
# No statistics, but detecions are visualized including dist for each frame

import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
from utils.plots import plot_one_box
from utils.datasets import letterbox
from utils.general import non_max_suppression
import yaml

# Define the input image size and the model input shape
input_size = (1024, 1024)  # Adjust this based on your model's expected input size

# Function to post-process the outputs
def postprocess_predictions(preds, conf_threshold=0.35, iou_threshold = 0.45):
    output_tensor = torch.from_numpy(preds)
    output_tensor = output_tensor.unsqueeze(0)

    nms_results = non_max_suppression(output_tensor, conf_thres=conf_threshold, iou_thres=iou_threshold)

    filtered_outputs = np.asarray([output.numpy() for output in nms_results if output.size(0) > 0]) # only return anchor tensors that still remain
    return filtered_outputs.squeeze(0) if np.size(filtered_outputs) >0 else [] # list of BB predictions including dist for each image (n,7) - [xyxy, conf, cls, dist]

# Run inference on an image
def run_inference(session, image_path, conf_threshold, iou_threshold):
    #img_resized, original_shape = preprocess_image(image_path, input_size)
    img = cv2.imread(image_path)
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=(1024, 1024), auto=False, stride=32, scaleup=False)  # Resize with padding
    img = img_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = img[:,:,::-1].transpose(2, 0, 1).copy()  # BGR to RGB, shape to [3, height, width]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    inputs = {session.get_inputs()[0].name: img}
    outname = [i.name for i in session.get_outputs()]

    # Run inference
    pred = session.run(outname, inputs)[0]  # Get predictions for batch or single image
    nms_results = postprocess_predictions(pred, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    return img_resized, nms_results

def getPathToFiles(yamlPath):
    with open(yamlPath, 'r') as f:
        data = yaml.safe_load(f)
        if os.path.isabs(data['test']):
            labels_path = os.path.join(os.path.dirname(data['test']), 'labels')
            return (data['test'], labels_path)
        else:
            images_path = os.path.join(os.path.dirname(yamlPath), data['test'])
            labels_path = os.path.join(os.path.dirname(images_path), 'labels')
            return(images_path, labels_path)

# Example usage
if __name__ == "__main__":
    test_dir = '/home/marten/Uni/Semester_4/src/DistanceEstimator/Dataset/Images/data.yaml'
    onnx_model_path = '/home/marten/Uni/Semester_4/src/YOLOv7-DL23/best.onnx'
    target_folder = "./testrun"
    device = 'cpu'
    visualize = True
    conf = 0.001
    iou = 0.65

    # Load Data
    imagesDir, labelsDir = getPathToFiles(test_dir)
    img_list = map(lambda x: os.path.join(imagesDir, x), sorted(os.listdir(imagesDir)))
    labels_list = map(lambda x: os.path.join(labelsDir, x), sorted(os.listdir(labelsDir)))

    # Load the ONNX model
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device == 'cuda':
        providers = ['CUDAExecutionProvider']
    else:
        raise ValueError("Device undefined")
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
 
    for image in img_list:
        # run ingerence on images with onnx model
        img_resized, results = run_inference(session, image, conf, iou)
        print(f"Image: {image} \t Detections: {np.shape(results)[0]}")
        # Optionally, visualize the results
        if visualize:
            # img = cv2.imread(image_path)
            for i, BB in enumerate(results):  # iterate over bounding boxes
                annotation = f"{BB[4]:.2f}, {BB[-1]:.3f}"
                # only plot box if conf > 0.25 -> same as in original yolo test script
                if BB[4] > 0.25:
                    plot_one_box(BB[0:4], img_resized, label=annotation)
            filename = os.path.basename(image)
            cv2.imwrite(os.path.join(target_folder, filename), img_resized)
