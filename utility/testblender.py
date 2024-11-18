import bpy
import threading
import time
import math

# Initial positions of the ship and buoy
ship_position = [0, 0, 0]
buoy_position = [2, 2, 0]

# Clear the existing scene (optional)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create the ship (using a cube as a placeholder)
bpy.ops.mesh.primitive_cube_add(scale=(1, 2, 0.5), location=(0, 0, 0))
ship = bpy.context.object
ship.name = 'Ship'

# Create the buoy (using a sphere)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(2, 2, 0))
buoy = bpy.context.object
buoy.name = 'Buoy'

# Create a camera
bpy.ops.object.camera_add(location=(0, -10, 5))
camera = bpy.context.object
camera.rotation_euler = (math.radians(60), 0, 0)
bpy.context.scene.camera = camera

# Create a light
bpy.ops.object.light_add(type='POINT', location=(10, -10, 10))
light = bpy.context.object
light.data.energy = 1000

# Function to update the positions in the background
def update_positions():
    global ship_position, buoy_position
    while True:
        # Simulate some data processing and update positions
        ship_position[0] += 0.1  # Move ship along x-axis
        buoy_position[1] += 0.05  # Move buoy along y-axis

        # Update the position of the ship and buoy in Blender
        ship.location = ship_position
        buoy.location = buoy_position
        
        time.sleep(0.1)  # Simulate delay in data processing

# Start the data update thread
threading.Thread(target=update_positions, daemon=True).start()

# Keep the Blender window running
bpy.context.window_manager.event_timer_add(0.1, window=bpy.context.window)  # This creates a real-time update loop in Blender

