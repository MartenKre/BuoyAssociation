import os

def filter_labels(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(file_path, 'w') as file:
                for line in lines:
                    if line.startswith('0 ') or line.startswith('1 '):
                        file.write(line)

def main():
    train_labels_dir = 'C:/Users/ben93/Downloads/CombinedDatasetsChallenge/CombinedDatasetsChallenge/labels/train'
    val_labels_dir = 'C:/Users/ben93/Downloads/CombinedDatasetsChallenge/CombinedDatasetsChallenge/labels/val'

    filter_labels(train_labels_dir)
    filter_labels(val_labels_dir)

    print("Label files filtered.")

if __name__ == "__main__":
    main()
