import cv2
import numpy as np

# Set up parameters
num_frames = 120  # Total number of frames in the video
resolution = np.array([1920, 1080])
fps = 30  # Frames per second

# Define constants needed for the transformations
f = 2.75  # focal length
#sensor_width = 36  # sensor width in mm
#sensor_height = 20.25  # sensor height in mm (assumed for 16:9 aspect ratio)
pixel_size = 0.00155 # 1.55μm

# Scaling factors for projection
alpha_u = 1 / (2*pixel_size)	# 2*pixel_size, since 4k is scaled down to FullHD
alpha_v = 1 / (2*pixel_size)	# otherwise this would be 1/pixel_size

# Original camera and point setup
t_updated_position = np.array([0, 0, 0])
T_CW = np.eye(4)
T_CW[:3, 3] = t_updated_position

P_w_camera_far_point = np.array([0, 500, 0, 1])
P_w_camera_second_point = np.array([100, 500, 0, 1])

# Initialize video writer
video_path = "attitude_sine_wave_transition.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_path, fourcc, fps, (resolution[0], resolution[1]))

# Function to create and save frames
def create_frame(roll, pitch):
    # Convert angles to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)

    # Update the rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    # Combined rotation matrix with updated roll and pitch
    R = R_y @ R_x

    # Update the transformation matrix
    T_CW[:3, :3] = R

    # Transform the updated points to camera coordinates
    P_c1 = np.linalg.inv(T_CW) @ P_w_camera_far_point
    P_c2 = np.linalg.inv(T_CW) @ P_w_camera_second_point

    # Project the updated points onto the image plane
    x_proj1 = (f * P_c1[0]) / -P_c1[2]
    y_proj1 = (f * -P_c1[1]) / -P_c1[2]

    x_proj2 = (f * P_c2[0]) / -P_c2[2]
    y_proj2 = (f * -P_c2[1]) / -P_c2[2]
    
    #print(x_proj1, y_proj1)
    #print(x_proj2, y_proj2)

    # Convert to pixel coordinates
    u1 = int(x_proj1 * alpha_u) + resolution[0]/2
    v1 = int(y_proj1 * alpha_v) + resolution[1]/2

    u2 = int(x_proj2 * alpha_u) + resolution[0]/2
    v2 = int(y_proj2 * alpha_v) + resolution[1]/2
    
    #print(u1, v1)
    #print(u2, v2)

    # Calculate the slope of the line (in image space) with the updated attitude
    slope = (v2 - v1) / (u2 - u1 + 0.000001)

    # Define the line extension to the image boundaries with the updated attitude
    u_start = 0
    u_end = resolution[0]
    v_start = int(v1 + slope * (u_start - u1))
    v_end = int(v1 + slope * (u_end - u1))

    # Create a blank image
    img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    # Draw the points and the extended line
    cv2.circle(img, (int(u1), int(v1)), 10, (0, 0, 255), -1)  # Red points
    cv2.circle(img, (int(u2), int(v2)), 10, (0, 0, 255), -1)
    cv2.line(img, (u_start, v_start), (u_end, v_end), (255, 0, 0), 2)  # Blue line

    # Add text showing roll and pitch values
    cv2.putText(img, f'Roll={roll:.2f}°, Pitch={pitch:.2f}°', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #print(u2)
    print(u1)
    # Write the frame to the video
    video.write(img)

# Generate frames with sine wave transition for roll and pitch
for frame_num in range(num_frames):
    t = frame_num / num_frames * 2 * np.pi  # Scale t to go from 0 to 2*pi
    roll = 90 - 30 * np.sin(t)
    pitch = 0 #+ 30 * np.sin(t)
    create_frame(roll, pitch)

# Release the video writer
video.release()

print(f"Video saved as {video_path}")
