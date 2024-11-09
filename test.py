import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# plot ship
#ax.plot3D(0, 0, 0, 'o', color = 'blue')

fig = plt.figure()
ax = plt.axes(projection="3d")
# Define arrow position and direction in the XY plane
scaling = 30
coords = np.asarray([[0,0,0], [-3,1, 0], [-2,0,0], [-3,-1,0], [0,0,0]])
coords *= scaling

poly = Poly3DCollection([coords], color=[(0,0,0.9)], edgecolor='k')
ax.add_collection3d(poly)

# create x,y
xx, yy = np.meshgrid(range(-100,700), range(-400,400))
z = xx*yy*0 -0.10
#ax.plot_surface(xx, yy, z, alpha=1, linewidth=0)

ax.plot3D(250, -80, 0,'o', color='red', alpha=1)

# Customize the view
ax.set_xlim(-100, 700)
ax.set_ylim(-400, 400)
ax.set_zlim(-1, 1)  # Keep it flat in the Z-axis for an XY view

# Optional: Labels for clarity
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=45, azim=-180)  # Set view to make it look like a 2D XY plane

plt.show()
