import json
import os
from pathlib import Path
from PIL import Image

def yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]

def convert_yolo_labels_to_coco(images_dir, labels_dir, output_json_path, categories):
    annotations = []
    images = []
    annotation_id = 1

    for image_path in Path(images_dir).glob("*.png"):
        image_id = image_path.stem
        img_width, img_height = Image.open(image_path).size
        images.append({"id": image_id, "width": img_width, "height": img_height, "file_name": image_path.name})

        label_file = Path(labels_dir) / (image_path.stem + ".txt")
        if label_file.exists():
            with open(label_file, 'r') as file:
                lines = file.readlines()

            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.split())
                bbox = yolo_to_coco(x_center, y_center, width, height, img_width, img_height)

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(categories)]
    }

    with open(output_json_path, 'w') as file:
        json.dump(coco_format, file, indent=4)

# images_dir = "C:/Users/ben93/Downloads/CombinedDatasetsChallenge/CombinedDatasetsChallenge/images/val"
# labels_dir = "C:/Users/ben93/Downloads/CombinedDatasetsChallenge/CombinedDatasetsChallenge/labels/val"
# output_json_path = "C:/Users/ben93/Downloads/CombinedDatasetsChallenge/CombinedDatasetsChallenge/val_annotations.json"
images_dir = r"C:\Users\ben93\Downloads\AllExceptCoco12\CombinedDatasets11\images"
labels_dir = r"C:\Users\ben93\Downloads\AllExceptCoco12\CombinedDatasets11\labels"
output_json_path = r"C:\Users\ben93\Downloads\AllExceptCoco12\CombinedDatasets11\labels.json"

categories = ["boat", "buoy"]  # Categories

convert_yolo_labels_to_coco(images_dir, labels_dir, output_json_path, categories)
