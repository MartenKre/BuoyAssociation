import math
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            category = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            distance = float(parts[5])
            data.append((category, x_center, y_center, width, height, distance))
    return data


def plot_top_down_view(data, fov_horizontal=50, image_width=1920, image_height=1080,
                       output_path='top_down_view_distances_with_cone.png'):
    fov_horizontal_rad = math.radians(fov_horizontal)
    positions = []

    for entry in data:
        category, x_center, y_center, width, height, distance = entry
        angle = (x_center - 0.5) * fov_horizontal_rad  # convert to radians and center
        x_pos = distance * math.tan(angle)  # x position in meters
        positions.append((x_pos, distance, category))

    positions_df = pd.DataFrame(positions, columns=['X Position (meters)', 'Distance (meters)', 'Category'])

    max_distance = max(positions_df['Distance (meters)'])
    left_boundary_x = max_distance * math.tan(-fov_horizontal_rad / 2)
    right_boundary_x = max_distance * math.tan(fov_horizontal_rad / 2)

    plt.figure(figsize=(12, 8))
    colors = {0: 'blue', 1: 'orange'}
    plt.scatter(positions_df['X Position (meters)'], positions_df['Distance (meters)'],
                c=positions_df['Category'].apply(lambda x: colors[x]), alpha=0.6)

    plt.plot([0, left_boundary_x], [0, max_distance], color='gray', linestyle='--', linewidth=1)
    plt.plot([0, right_boundary_x], [0, max_distance], color='gray', linestyle='--', linewidth=1)

    plt.title('Top-Down View of Object Distances with Image Cone Boundaries')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Distance (meters)')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# Example usage:
# file_path = r"C:\Users\ben93\Downloads\labels\labels\bd9c3df4-a53d-4587-ae3c-a330340b04e1_79831.txt"
# file_path = r"C:\Users\ben93\Downloads\labels\labels\bd9c3df4-a53d-4587-ae3c-a330340b04e1_79832.txt"
file_paths = r"C:\Users\ben93\Downloads\labels\labels"
list_of_txts = os.listdir(file_paths)
for j in range(10):
    data = parse_txt(os.path.join(file_paths,list_of_txts[j]))
    plot_top_down_view(data)
