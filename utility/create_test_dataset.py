
"""
Script to create Dataset in Yolo Format to Test Buoy Matching 
Test data contains BBs with buoy id and a file containing the ships imu data for each file
"""

import os
import json
from os.path import isdir
import numpy as np
import shutil
from GeoData import GetGeoData

def labelsJSON2Yolo(labels):
    # fuction converts json data to yolo format with corresponding buoy ID
    result = []
    for BB in labels[1]["objects"]:
        if "distanceGT" in BB["attributes"]:
            # get query ID
            lat_BB = BB["attributes"]["buoyLat"]
            lng_BB = BB["attributes"]["buoyLng"]

            id = buoyGTData.getBuoyID(lat_BB, lng_BB)
            if id == 0 or id is None or id == "null":
                raise ValueError("Invalid Buoy ID encountered")

            # get BB info in yolo format
            x1 = BB["x1"]
            y1 = BB["y1"]
            x2 = BB["x2"]
            y2 = BB["y2"]
            bbCenterX = ((x1 + x2) / 2) / 1920 
            bbCenterY = ((y1 + y2) / 2) / 1080 
            bbWidth = (x2-x1) / 1920 
            bbHeight = (y2-y1) / 1080 
            bbInfo = str(bbCenterX) + " " + str(bbCenterY) + " " + str(bbWidth) + " " + str(bbHeight) + " " + id + "\n"

            result.append(bbInfo)

    return result 

def getIMUData(path):
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


# Settings:
verbose = True
target_dir = "/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/YOLO_Testset"
data_path = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/"
if os.path.exists(target_dir):
    raise ValueError("Aborting... Specified target dir already exists:", target_dir)
os.makedirs(target_dir, exist_ok=True)

buoyGTData = GetGeoData()

os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)
imu_test_file = os.path.join(target_dir, 'img_data' + '.json')
imu_dict = {}
sample_counter = 0

datafolders = []
for folder in os.listdir(data_path):
    if os.path.isdir(os.path.join(data_path, folder)):
        if folder == 'Testdata':
            datafolders.append(folder)

print("Folders Found: ", sorted(datafolders))

for folder in datafolders:
    parent_folder = os.path.join(data_path, folder)
    for subfolder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, subfolder)
        print("Processing: " , folder_path)
        images = os.path.join(folder_path, "images")
        imu = os.listdir(os.path.join(folder_path, "imu"))[0]
        labels = os.path.join(folder_path, "labels")
        imu_data = getIMUData(os.path.join(folder_path, "imu", imu)) 

        for sample in os.listdir(images):
            # copy image
            src_path_img = os.path.join(images, sample)
            sample_name = "0" * (5-len(list(str(sample_counter)))) + str(sample_counter)
            dest_path = os.path.join(target_dir, "images", sample_name + '.png')

            # extract imu data
            frame_id = int(sample.split(".")[0]) - 1
            imu_curr = imu_data[frame_id] 
            ship_pose = [imu_curr[3],imu_curr[4],imu_curr[2]]
            imu_dict[sample_name] = ship_pose

            # create labels file
            src_path = os.path.join(labels, sample+".json")
            label_data = json.load(open(src_path, 'r'))
            txtlabels = labelsJSON2Yolo(label_data, ship_pose, buoyGTData)
            if txtlabels is None: # if dist between label buoy and query buoy too large -> skip 
                continue
            if len(txtlabels) == 0: # if labels file empty
                if verbose:
                    print(f"\t \t Warning: Empty labels file: {src_path}")
                #continue
            labelfile = os.path.join(target_dir, "labels", sample_name + '.txt')


            # save data
            shutil.copy(src_path_img, dest_path)
            
            with open(labelfile, 'w') as f:
                f.writelines(txtlabels)
            sample_counter += 1

# save imu data as json
with open(imu_test_file, "w") as f:
    json.dump(imu_dict, f)

print("DONE!")
print("Total Processed: ", datafolders)
