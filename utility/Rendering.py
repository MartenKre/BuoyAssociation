from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
from Transformations import T_W_Ship, LatLng2ECEF, T_ECEF_Ship
import math

class RenderBuoys():
    def __init__(self):
        self.window_size = [1920, 1080]
        self.window_scaling_factor = 0.5

        # camera params
        self.fov = 60
        self.near_plane = 0
        self.far_plane = 5000 
        self.follow_ship = True

        #ship params
        self.ship_scaling = 50

        # logic function
        self.logic_interval = 30    # update logic func every 30 ms

        #data
        # ship data
        ship_initPos = np.array([0, 0, 0])  #rm
        self.ship_initHeading = np.radians(0)   #rm
        self.ship_path = [ship_initPos] #rm
        #self.ship_path = []

        # buoy preds
        self.preds=[[300, 50]] #rm
        # self.preds=[]

        # buoy GT data
        self.buoysGT = [[250, 40]] #rm
        # self.buoysGT = []

        # transformation matrices
        self.W_T_Ship = T_W_Ship(ship_initPos, self.ship_initHeading)
        self.ECEF_T_W = T_ECEF_Ship(*ship_initPos, 0)    # T matrix to transform ecef coords into world cs (e.g buoys)

        # time
        self.t = 0
        self.dt = 0.03  # delta for logic function

        # OPENGL Init
        glutInit() # Initialize a glut instance which will allow us to customize our window
        glutInitDisplayMode(GLUT_RGBA) # Set the display mode to be colored
        glutInitWindowSize(int(self.window_size[0]*self.window_scaling_factor), int(self.window_size[1]*self.window_scaling_factor))   # Set the width and height of your window
        glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
        wind = glutCreateWindow("Buoy Predictions") # Give your window a title
        glutDisplayFunc(self.showScreen)  # Tell OpenGL to call the showScreen method continuously
        glutIdleFunc(self.showScreen)     # Draw any graphics or shapes in the showScreen function at all times
        glutTimerFunc(self.logic_interval, self.updateLogic, 0)  # Start the logic timer
        glutMainLoop()  # Keeps the window created above displaying/running in a loop

    def initTransformations(self, lat, lng, heading):
        # function needs to be called to initialize Transformation matrices
        x, y, z = LatLng2ECEF(lat, lng)
        self.ECEF_T_W = T_ECEF_Ship(x, y, z, 0)
        self.W_T_Ship = T_W_Ship(np.array[0, 0, 0], heading)
        self.ship_initHeading = np.radians(heading)
        self.ship_path.append(np.array([0, 0, 0]))

    def setPreds(self, preds):
        # list of buoy preds (lat&lng coords)
        self.preds = []
        for pred in preds:
            x,y,z = LatLng2ECEF(pred[0], pred[1])
            p_buoy = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])  # buoy pred in ship CS
            self.preds.append([p_buoy[0], p_buoy[1]])
    
    def setShipData(self, lat, lng, heading):
        # current pos and heading of ship
        x, y, z = LatLng2ECEF(lat, lng)
        p_WCS = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])
        self.ship_path.append(p_WCS[0:3])
        self.W_T_Ship = T_W_Ship(p_WCS[:3], heading)

    def setBuoyGT(self, buoysGT):
        # list of buoy gt data (lat & lng coords)
        self.buoysGT = []
        for buoy in buoysGT:
            x,y,z = LatLng2ECEF(buoy[0], buoy[1])
            p_buoy = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])  # buoy pred in ship CS
            self.preds.append([p_buoy[0], p_buoy[1]])
        
    def renderShipIcon(self):
        # render ship icon
        glColor3f(0.0, 0.0, 0.7)
        # Rotate ship icon based on its location in World CS, sinde polygon is defined in ship CS
        p1 = self.W_T_Ship@(np.array([1, 0, 0, 1/self.ship_scaling])*self.ship_scaling)
        p2 = self.W_T_Ship@(np.array([-2, -1, 0, 1/self.ship_scaling])*self.ship_scaling)
        p3 = self.W_T_Ship@(np.array([-1, 0, 0, 1/self.ship_scaling])*self.ship_scaling)
        p4 = self.W_T_Ship@(np.array([-2, 1, 0, 1/self.ship_scaling])*self.ship_scaling)

        glBegin(GL_POLYGON)
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p3[0], p3[1], p3[2])
        glVertex3f(p4[0], p4[1], p4[2])
        glVertex3f(p1[0], p1[1], p1[2]) 
        glEnd()

    def renderShipPath(self, dashed=False):
        # render ship path
        glColor3f(0.8, 0.8, 0.0)

        if not dashed:
            glLineWidth(3)
            glBegin(GL_LINE_STRIP)
            for p in self.ship_path:
                glVertex3f(p[0], p[1], p[2])
            glEnd()
        else:
            glLineWidth(5)
            glBegin(GL_LINES)
            for i, p in enumerate(self.ship_path):
                if i % 10==0:
                    glVertex3f(p[0], p[1], p[2])
            glEnd()

    def renderBuoyGT(self):
        # the predictions are expected to be in lat lon coords
        for x, y in self.buoysGT:
            glPushMatrix()
            glTranslatef(x, y, 0.0) 
            glColor3f(0.0, 0.8, 0.0)  # Set color to red
            glutSolidSphere(10, 50, 50)  # Draw a sphere with radius, 50 slices, and 50 stacks
            glPopMatrix()
            ## draw line from ship to buoy
            p_ship = self.W_T_Ship[:3, 3]
            glColor3f(0.5, 0.5, 0.5)  # Set color to red
            glBegin(GL_LINES)
            a = np.array([x, y, 0]) - p_ship
            points = np.asarray([a*x/(np.linalg.norm(a)/10)+p_ship for x in range(0, int(np.linalg.norm(a)/10))])
            for point in points:
                glVertex3f(point[0], point[1], 0)
            glEnd()

    def renderBuoyPreds(self):
        # the predictions are expected to be in lat lon coords
        for x, y in self.preds:
            glPushMatrix()
            glTranslatef(x, y, 0.0)  # Move the sphere to (2, 1, 0)
            glColor3f(0.8, 0, 0.0)  # Set color to red
            glutSolidSphere(10, 50, 50)  # Draw a sphere with radius, 50 slices, and 50 stacks
            glPopMatrix()
            ## draw line from ship to buoy
            p_ship = self.W_T_Ship[:3, 3]
            glColor3f(0.5, 0.5, 0.5)  # Set color to red
            glBegin(GL_LINES)
            a = np.array([x, y, 0]) - p_ship
            points = np.asarray([a*x/(np.linalg.norm(a)/10)+p_ship for x in range(0, int(np.linalg.norm(a)/10))])
            for point in points:
                glVertex3f(point[0], point[1], 0)
            glEnd()


    def renderSurface(self):
            glBegin(GL_QUADS)  # Draw a flat surface as a quad (rectangle)
            glColor4f(0.0, 0.5, 1.0, 0.5)  # Set water color (light blue)

            # Define the vertices for the flat surface
            glVertex3f(-10000.0, 10000.0, 0.0)  # Bottom-left
            glVertex3f(-10000.0, -10000.0, 0.0)  # Bottom-right
            glVertex3f(10000.0, -10000.0,  0.0)  # Top-right
            glVertex3f(10000.0, 10000.0,  0.0)  # Top-left

            glEnd()

    def moveShip(self):
        new_heading = np.radians(np.sin(0.3*self.t) * 45) + self.ship_initHeading
        pos_new_WCS= self.W_T_Ship @ np.array([20*self.dt, 0, 0, 1])
        self.W_T_Ship = T_W_Ship(pos_new_WCS[:3], new_heading)
        self.ship_path.append(pos_new_WCS[0:3])

    def updateLogic(self, value):
        # logic function, called every 30 ms
        self.t += self.dt
        self.moveShip()

        # Schedule the next logic update
        glutTimerFunc(self.logic_interval, self.updateLogic, 0)

    def Camera(self):
        # cam at World CS pos & heading of 0
        gluLookAt(-500, 0, 200, 10, 0, -3, 0, 0, 1) #pos_xyz, view_vector xyz, up axis

    def shipCam(self):
        # cam that follows ship
        pos = np.array([-400, 0, 200, 1])  # camera pos in ship cs
        view_vec = np.array([500, 0, 0, 1])
        z = np.array([0, 0, 1])
        pos = self.W_T_Ship@pos
        view_vec = self.W_T_Ship@view_vec
        gluLookAt(*pos[:3], *view_vec[:3], 0, 0, 1) #pos_xyz, view_vector xyz, up axis

    def iterate(self):
        size = np.ndarray.tolist(glGetIntegerv(GL_VIEWPORT))
        glViewport(0, 0, size[2], size[3])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if size[3] != 0:
            aspect_ratio = size[2] / size[3]    # screen width / height
        else:
            aspect_ratio = 16/9
        near_plane = self.near_plane  # dont show objects closer that this
        far_plane = self.far_plane    # dont show objects further than this
        gluPerspective(self.fov, aspect_ratio, near_plane, far_plane)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity()
        if self.follow_ship == False:
            self.Camera()
        else:
            self.shipCam()

    def showScreen(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glClearColor(0.7, 0.7, 0.7, 1.0)  
        glEnable(GL_LINE_SMOOTH)  # Enable line smoothing for better appearance
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable( GL_BLEND )
        self.iterate()
        self.renderSurface()
        self.renderShipIcon()
        self.renderShipPath()
        self.renderBuoyPreds()
        self.renderBuoyGT()
        glutSwapBuffers()


render = RenderBuoys()