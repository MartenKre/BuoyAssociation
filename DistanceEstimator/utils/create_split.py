import os
import random
import shutil

def get_file_basename(file):
    return os.path.splitext(file)[0]

def get_extension(file):
    return os.path.splitext(file)[1]

def main():
    images_dir = 'C:/Users/ben93/Downloads/CombinedDatasetsChallenge/CombinedDatasetsChallenge/images'
    labels_dir = 'C:/Users/ben93/Downloads/CombinedDatasetsChallenge/CombinedDatasetsChallenge/labels'

    # Get list of all images and labels
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    label_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))]

    # Store image files with their extensions
    image_files_with_ext = {get_file_basename(f): get_extension(f) for f in image_files}

    # Filter out files without a corresponding pair
    image_files = set(image_files_with_ext.keys())
    label_files = {get_file_basename(f) for f in label_files}
    valid_files = image_files.intersection(label_files)

    # Remove unpaired files
    for f in image_files - valid_files:
        os.remove(os.path.join(images_dir, f + image_files_with_ext[f]))
    for f in label_files - valid_files:
        os.remove(os.path.join(labels_dir, f + '.txt'))

    # Split into train and validation sets
    valid_files = list(valid_files)
    random.shuffle(valid_files)
    split_index = int(0.8 * len(valid_files))

    train_files = valid_files[:split_index]
    val_files = valid_files[split_index:]

    # Create train and val directories
    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_labels_dir = os.path.join(labels_dir, 'val')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Move files to respective directories
    for f in train_files:
        shutil.move(os.path.join(images_dir, f + image_files_with_ext[f]), train_images_dir)
        shutil.move(os.path.join(labels_dir, f + '.txt'), train_labels_dir)

    for f in val_files:
        shutil.move(os.path.join(images_dir, f + image_files_with_ext[f]), val_images_dir)
        shutil.move(os.path.join(labels_dir, f + '.txt'), val_labels_dir)

    print(f"Training set: {len(train_files)} pairs")
    print(f"Validation set: {len(val_files)} pairs")

if __name__ == "__main__":
    main()
