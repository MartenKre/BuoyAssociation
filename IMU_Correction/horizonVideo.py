import cv2
import numpy as np
import os

path_to_video = "/home/marten/Uni/Semester_4/src/BuoyAssociation/IMU_Correction/17_2.avi"
path_to_imu = "/home/marten/Uni/Semester_4/src/BuoyAssociation/IMU_Correction/furuno_17.txt"
out_filename = os.path.basename(path_to_video).replace(".avi", ".mp4")
out_path = f"/home/marten/Uni/Semester_4/src/BuoyAssociation/IMU_Correction/{out_filename}"

#####################
### IMU (Ship) CS: x points to front, y to left, z to top
### Camera CS: x points to right, y to bottom, z to front
### FROM Ship to Camera: Roll: -90, Pitch: 90
#####################

# Set up parameters
resolution = np.array([1920, 1080])

# Define constants needed for the transformations
f = 2.75  # focal length
pixel_size = 0.00155 # 1.55μm

# Scaling factors for projection
alpha_u = 1 / (2*pixel_size)	# 2*pixel_size, since 4k is scaled down to FullHD
alpha_v = 1 / (2*pixel_size)	# otherwise this would be 1/pixel_size

# Original camera and point setup
t_updated_position = np.array([0, 0, 2])
T_WC = np.eye(4)
T_WC[:3, 3] = t_updated_position

# horizon points -> in global CS, used to draw 
P_w_camera_far_point = np.array([5000, -1000, 0, 1])
P_w_camera_second_point = np.array([5000, 1000, 0, 1])

def draw_horizon_on_video(video_path, output_path, imu_file):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the video writer with the same properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi or 'mp4v' for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read the horizon positions from the text file
    with open(imu_file, 'r') as f:
        imu_data = [line.split(",") for line in f.readlines()]

    # Make sure we have horizon data for each frame
    if len(imu_data) < total_frames:
        print("Warning: Horizon positions are fewer than total frames. Some frames will not have a line drawn.")
    
    # Process each frame
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing {frame_index} / {total_frames}")

        # Draw the horizon line if there is data for this frame
        if frame_index < len(imu_data):
            roll = -1* float(imu_data[frame_index][0])
            pitch = float(imu_data[frame_index][1])
            # compute horizon points in ship cs (rotated by roll and pitch from IMU)
            p1, p2 = global_to_IMU(roll, pitch)
            # transform hz points from ship to camera cs and get horizon line
            v1, v2 = get_horizon(p1, p2)
            color = (0, 255, 0)  # Green color for the horizon line
            thickness = 2
            cv2.line(frame, (0, v1), (frame_width, v2), color, thickness)
            cv2.putText(frame, f'Roll={roll:.2f}, Pitch={pitch:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the modified frame to the output video
        out.write(frame)

        frame_index += 1

    # Release video objects
    cap.release()
    out.release()
    print("Video processing complete. Output saved as:", output_path)

def global_to_IMU(roll, pitch):
    # transforms a point from global to IMU (ship)
    # global is equal to IMU CS but always has 0 pitch & roll
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    # Combined rotation matrix with updated roll and pitch
    R_WS = R_y @ R_x

    T_WS = np.eye(4)
    T_WS[:3,:3] = R_WS
    P_s1 = np.linalg.inv(T_WS) @ P_w_camera_far_point
    P_s2 = np.linalg.inv(T_WS) @ P_w_camera_second_point

    return P_s1, P_s2

def get_horizon(p1, p2):
    # Function to get pixel coordinates for horizon line

    # First transform points from IMU (ship) CS to Camera
    # Use fixed rotation matrix
    # reihenfolge: yaw - pitch - roll um from Ship zu Camera zu kommen
    roll_rad = np.radians(-90)
    pitch_rad = np.radians(0)
    yaw_rad = np.radians(-90)

    # Update the rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    # Combined rotation matrix with updated roll and pitch
    R = R_z @ R_y @ R_x

    # Update the transformation matrix
    T_WC[:3, :3] = R

    # Transform the updated points to camera coordinates
    P_c1 = np.linalg.inv(T_WC) @ p1
    P_c2 = np.linalg.inv(T_WC) @ p2

    # Project the updated points onto the image plane
    x_proj1 = (f * P_c1[0]) / P_c1[2]
    y_proj1 = (f * P_c1[1]) / P_c1[2]

    x_proj2 = (f * P_c2[0]) / P_c2[2]
    y_proj2 = (f * P_c2[1]) / P_c2[2]

    # Convert to pixel coordinates
    u1 = int(x_proj1 * alpha_u) + resolution[0]/2
    v1 = int(y_proj1 * alpha_v) + resolution[1]/2

    u2 = int(x_proj2 * alpha_u) + resolution[0]/2
    v2 = int(y_proj2 * alpha_v) + resolution[1]/2

    # Calculate the slope of the line (in image space) with the updated attitude
    slope = (v2 - v1) / (u2 - u1 + 0.000001)

    # Define the line extension to the image boundaries with the updated attitude
    u_start = 0
    u_end = resolution[0]
    v_start = int(v1 + slope * (u_start - u1))
    v_end = int(v1 + slope * (u_end - u1))

    return(v_start, v_end)

draw_horizon_on_video(path_to_video, out_path, path_to_imu)