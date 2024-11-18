# from OpenGL.GL import *
# import OpenGL.GL.shaders
# from OpenGL.GLUT import *
# from OpenGL.GLU import *
# import numpy as np
# from Transformations import T_W_Ship, LatLng2ECEF, T_ECEF_Ship
# import math

# class RenderBuoys():
#     def __init__(self):
#         self.window_size = [1920, 1080]
#         self.window_scaling_factor = 0.5

#         # camera params
#         self.fov = 60
#         self.near_plane = 0
#         self.far_plane = 5000 
#         self.follow_ship = True

#         #ship params
#         self.ship_scaling = 50

#         # logic function
#         self.logic_interval = 30    # update logic func every 30 ms

#         #data
#         # ship data
#         ship_initPos = np.array([0, 0, 0])  #rm
#         self.ship_initHeading = np.radians(0)   #rm
#         self.ship_path = [ship_initPos] #rm
#         #self.ship_path = []

#         # buoy preds
#         self.preds=[[300, 50]] #rm
#         # self.preds=[]

#         # buoy GT data
#         self.buoysGT = [[250, 40]] #rm
#         # self.buoysGT = []

#         # transformation matrices
#         self.W_T_Ship = T_W_Ship(ship_initPos, self.ship_initHeading)
#         self.ECEF_T_W = T_ECEF_Ship(*ship_initPos, 0)    # T matrix to transform ecef coords into world cs (e.g buoys)

#         # time
#         self.t = 0
#         self.dt = 0.03  # delta for logic function

#         # OPENGL Init
#         glutInit() # Initialize a glut instance which will allow us to customize our window
#         glutInitDisplayMode(GLUT_RGBA) # Set the display mode to be colored
#         glutInitWindowSize(int(self.window_size[0]*self.window_scaling_factor), int(self.window_size[1]*self.window_scaling_factor))   # Set the width and height of your window
#         glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
#         wind = glutCreateWindow("Buoy Predictions") # Give your window a title
#         glutDisplayFunc(self.showScreen)  # Tell OpenGL to call the showScreen method continuously
#         glutIdleFunc(self.showScreen)     # Draw any graphics or shapes in the showScreen function at all times
#         glutTimerFunc(self.logic_interval, self.updateLogic, 0)  # Start the logic timer
#         glutMainLoop()  # Keeps the window created above displaying/running in a loop

#     def loadShaders(self, filename):
#         with open(filename, 'r') as file:
#             return file.read()

#     def compile_shader(self, vertex_source, fragment_source):
#         vertex_shader = glCreateShader(GL_VERTEX_SHADER)
#         glShaderSource(vertex_shader, vertex_source)
#         glCompileShader(vertex_shader)
#         if glGetShaderiv(vertex_shader, GL_COMPILE_STATUS) != GL_TRUE:
#             raise RuntimeError(glGetShaderInfoLog(vertex_shader).decode('utf-8'))
        
#         fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
#         glShaderSource(fragment_shader, fragment_source)
#         glCompileShader(fragment_shader)
#         if glGetShaderiv(fragment_shader, GL_COMPILE_STATUS) != GL_TRUE:
#             raise RuntimeError(glGetShaderInfoLog(fragment_shader).decode('utf-8'))
        
#         shader_program = glCreateProgram()
#         glAttachShader(shader_program, vertex_shader)
#         glAttachShader(shader_program, fragment_shader)
#         glLinkProgram(shader_program)
#         if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
#             raise RuntimeError(glGetProgramInfoLog(shader_program).decode('utf-8'))
        
#         # Clean up shaders as they are now linked into the program
#         glDeleteShader(vertex_shader)
#         glDeleteShader(fragment_shader)
        
#         return shader_program


#     def initTransformations(self, lat, lng, heading):
#         # function needs to be called to initialize Transformation matrices
#         x, y, z = LatLng2ECEF(lat, lng)
#         self.ECEF_T_W = T_ECEF_Ship(x, y, z, 0)
#         self.W_T_Ship = T_W_Ship(np.array[0, 0, 0], heading)
#         self.ship_initHeading = np.radians(heading)
#         self.ship_path.append(np.array([0, 0, 0]))

#     def setPreds(self, preds):
#         # list of buoy preds (lat&lng coords)
#         self.preds = []
#         for pred in preds:
#             x,y,z = LatLng2ECEF(pred[0], pred[1])
#             p_buoy = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])  # buoy pred in ship CS
#             self.preds.append([p_buoy[0], p_buoy[1]])
    
#     def setShipData(self, lat, lng, heading):
#         # current pos and heading of ship
#         x, y, z = LatLng2ECEF(lat, lng)
#         p_WCS = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])
#         self.ship_path.append(p_WCS[0:3])
#         self.W_T_Ship = T_W_Ship(p_WCS[:3], heading)

#     def setBuoyGT(self, buoysGT):
#         # list of buoy gt data (lat & lng coords)
#         self.buoysGT = []
#         for buoy in buoysGT:
#             x,y,z = LatLng2ECEF(buoy[0], buoy[1])
#             p_buoy = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])  # buoy pred in ship CS
#             self.preds.append([p_buoy[0], p_buoy[1]])
        
#     def renderShipIcon(self):
#         # render ship icon
#         glColor3f(0.0, 0.0, 0.7)
#         # Rotate ship icon based on its location in World CS, sinde polygon is defined in ship CS
#         p1 = self.W_T_Ship@(np.array([1, 0, 0, 1/self.ship_scaling])*self.ship_scaling)
#         p2 = self.W_T_Ship@(np.array([-2, -1, 0, 1/self.ship_scaling])*self.ship_scaling)
#         p3 = self.W_T_Ship@(np.array([-1, 0, 0, 1/self.ship_scaling])*self.ship_scaling)
#         p4 = self.W_T_Ship@(np.array([-2, 1, 0, 1/self.ship_scaling])*self.ship_scaling)

#         glBegin(GL_POLYGON)
#         glVertex3f(p1[0], p1[1], p1[2])
#         glVertex3f(p2[0], p2[1], p2[2])
#         glVertex3f(p3[0], p3[1], p3[2])
#         glVertex3f(p4[0], p4[1], p4[2])
#         glVertex3f(p1[0], p1[1], p1[2]) 
#         glEnd()

#     def renderShipPath(self, dashed=False):
#         # render ship path
#         glColor3f(0.8, 0.8, 0.0)

#         if not dashed:
#             glLineWidth(3)
#             glBegin(GL_LINE_STRIP)
#             for p in self.ship_path:
#                 glVertex3f(p[0], p[1], p[2])
#             glEnd()
#         else:
#             glLineWidth(5)
#             glBegin(GL_LINES)
#             for i, p in enumerate(self.ship_path):
#                 if i % 10==0:
#                     glVertex3f(p[0], p[1], p[2])
#             glEnd()

#     def renderBuoyGT(self):
#         # the predictions are expected to be in lat lon coords
#         for x, y in self.buoysGT:
#             ## draw line from ship to buoy
#             p_ship = self.W_T_Ship[:3, 3]
#             glColor3f(0.5, 0.5, 0.5)  # Set color to red
#             glBegin(GL_LINES)
#             a = np.array([x, y, 0]) - p_ship
#             points = np.asarray([a*x/(np.linalg.norm(a)/10)+p_ship for x in range(0, int(np.linalg.norm(a)/10))])
#             for point in points:
#                 glVertex3f(point[0], point[1], 0)
#             glEnd()
#             glPushMatrix()
#             # render buoy
#             glTranslatef(x, y, 0.0) 
#             glColor3f(0.0, 0.8, 0.0)  # Set color to red
#             glutSolidSphere(10, 50, 50)  # Draw a sphere with radius, 50 slices, and 50 stacks
#             glPopMatrix()

#     def renderBuoyPreds(self):
#         # the predictions are expected to be in lat lon coords
#         for x, y in self.preds:
#             glPushMatrix()
#             glTranslatef(x, y, 0.0)  # Move the sphere to (2, 1, 0)
#             glColor3f(0.8, 0, 0.0)  # Set color to red
#             glutSolidSphere(10, 50, 50)  # Draw a sphere with radius, 50 slices, and 50 stacks
#             glPopMatrix()
#             ## draw line from ship to buoy
#             p_ship = self.W_T_Ship[:3, 3]
#             glColor3f(0.5, 0.5, 0.5)  # Set color to red
#             glBegin(GL_LINES)
#             a = np.array([x, y, 0]) - p_ship
#             points = np.asarray([a*x/(np.linalg.norm(a)/10)+p_ship for x in range(0, int(np.linalg.norm(a)/10))])
#             for point in points:
#                 glVertex3f(point[0], point[1], 0)
#             glEnd()


#     def renderSurface(self):
#             glBegin(GL_QUADS)  # Draw a flat surface as a quad (rectangle)
#             glColor4f(0.0, 0.5, 1.0, 0.5)  # Set water color (light blue)

#             # Define the vertices for the flat surface
#             glVertex3f(-10000.0, 10000.0, 0.0)  # Bottom-left
#             glVertex3f(-10000.0, -10000.0, 0.0)  # Bottom-right
#             glVertex3f(10000.0, -10000.0,  0.0)  # Top-right
#             glVertex3f(10000.0, 10000.0,  0.0)  # Top-left

#             glEnd()

#     def moveShip(self):
#         new_heading = np.radians(np.sin(0.3*self.t) * 45) + self.ship_initHeading
#         pos_new_WCS= self.W_T_Ship @ np.array([20*self.dt, 0, 0, 1])
#         self.W_T_Ship = T_W_Ship(pos_new_WCS[:3], new_heading)
#         self.ship_path.append(pos_new_WCS[0:3])

#     def updateLogic(self, value):
#         # logic function, called every 30 ms
#         self.t += self.dt
#         self.moveShip()

#         # Schedule the next logic update
#         glutTimerFunc(self.logic_interval, self.updateLogic, 0)

#     def Camera(self):
#         # cam at World CS pos & heading of 0
#         gluLookAt(-500, 0, 200, 10, 0, -3, 0, 0, 1) #pos_xyz, view_vector xyz, up axis

#     def shipCam(self):
#         # cam that follows ship
#         pos = np.array([-400, 0, 200, 1])  # camera pos in ship cs
#         view_vec = np.array([500, 0, 0, 1])
#         z = np.array([0, 0, 1])
#         pos = self.W_T_Ship@pos
#         view_vec = self.W_T_Ship@view_vec
#         gluLookAt(*pos[:3], *view_vec[:3], 0, 0, 1) #pos_xyz, view_vector xyz, up axis

#     def iterate(self):
#         size = np.ndarray.tolist(glGetIntegerv(GL_VIEWPORT))
#         glViewport(0, 0, size[2], size[3])
#         glMatrixMode(GL_PROJECTION)
#         glLoadIdentity()
#         if size[3] != 0:
#             aspect_ratio = size[2] / size[3]    # screen width / height
#         else:
#             aspect_ratio = 16/9
#         near_plane = self.near_plane  # dont show objects closer that this
#         far_plane = self.far_plane    # dont show objects further than this
#         gluPerspective(self.fov, aspect_ratio, near_plane, far_plane)
#         glMatrixMode (GL_MODELVIEW)
#         glLoadIdentity()
#         if self.follow_ship == False:
#             self.Camera()
#         else:
#             self.shipCam()

#     def showScreen(self):
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glLoadIdentity()
#         glClearColor(0.7, 0.7, 0.7, 1.0)  
#         glEnable(GL_LINE_SMOOTH)  # Enable line smoothing for better appearance
#         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
#         glEnable( GL_BLEND )
#         self.iterate()
#         self.renderSurface()
#         self.renderShipIcon()
#         self.renderShipPath()
#         self.renderBuoyPreds()
#         self.renderBuoyGT()
#         glutSwapBuffers()


# render = RenderBuoys()




from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, PointLight, AmbientLight
from panda3d.core import Vec3, LineSegs, NodePath
from panda3d.core import *
import numpy as np
from Transformations import T_W_Ship, LatLng2ECEF, T_ECEF_Ship, extract_RPY
import math
from copy import deepcopy
from direct.filter.CommonFilters import CommonFilters

class SimpleScene(ShowBase):
    def __init__(self):
        super().__init__()
        self.render.setShaderAuto()
        self.render.setAntialias(AntialiasAttrib.MMultisample)  # Enable anti-aliasing
        lens = base.cam.node().getLens()
        lens.setFar(3000)
        #filter = CommonFilters(self.win, self.cam)
        #filter.setBloom()

        # Disable the default camera trackball control
        self.disableMouse()

        self.set_background_color(0.71, 0.71, 0.71, 1)  # (R, G, B, A) values (0 to 1)

        self.follow_ship = True
        # data
        # ship data
        ship_initPos = np.array([0, 0, 0])  #rm
        self.ship_initHeading = np.radians(0)   #rm
        self.ship_path = [ship_initPos] #rm
        #self.ship_path = []

        # time
        self.t = 0
        self.dt = 0.03  # delta for logic function
        self.counter = 0

        # buoy preds
        self.preds=[[300, 50]] #rm
        # self.preds=[]

        # buoy GT data
        self.buoysGT = [[250, 40]] #rm
        # self.buoysGT = []

        # transformation matrices
        self.W_T_Ship = T_W_Ship(ship_initPos, self.ship_initHeading)
        self.ECEF_T_W = T_ECEF_Ship(*ship_initPos, 0)    # T matrix to transform ecef coords into world cs (e.g buoys)

        self.buoy_model_green = self.loader.loadModel("/home/marten/Uni/Semester_4/src/BuoyAssociation/utility/buoy_green.egg")
        self.buoy_model_red = self.loader.loadModel("/home/marten/Uni/Semester_4/src/BuoyAssociation/utility/buoy_red.egg")
        #self.light_model = self.loader.loadModel("/home/marten/Uni/Semester_4/src/BuoyAssociation/utility/test_rendering/sphere.bam")

        # render ship
        self.ship = self.loader.loadModel("/home/marten/Uni/Semester_4/src/BuoyAssociation/utility/ship.egg")  # Load a plane model
        self.ship_scale = 30
        self.ship.setScale(self.ship_scale, self.ship_scale, self.ship_scale)  # Scale the plane to make it larger
        self.ship.setPos(0, 0, 0)  # Position it at the origin
        self.ship.setHpr(0, 0, 0)
        self.ship.setShaderAuto()
        self.ship.reparentTo(self.render)  # Add to the scene

        self.pred_buoys_render = []
        self.gt_buoys_render = []
        self.buoy_lights = []
        self.line_node_path = None  # Store the node path for the line
        self.lines = LineSegs()  # Create LineSegs object once 
        self.lines.setColor(1, 0, 0, 1)  # (R, G, B, A) for red
        self.lines.set_thickness(3)
        self.floor = None

        # Add a point light
        plight = PointLight("plight")
        plight.setColor((0.3, 0.3, 0.3, 1))  # Slightly red light
        plight.attenuation = (0.3, 0, 0.00000005)
        plight.set_shadow_caster(True, 1024, 1024)  # Enable shadows with resolution
        self.plnp = self.render.attachNewNode(plight)
        self.plnp.setPos(0, 0, 100)  # Position of the point light
        self.render.setLight(self.plnp)

        # Add ambient light
        alight = AmbientLight("alight")
        alight.setColor((0.1, 0.1, 0.1, 1))  # Dim light
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        self.renderSurface()
        self.taskMgr.add(self.logic_loop, "logic_loop")

    def logic_loop(self, task):
        self.updateLogic()
        self.showScreen()

        return task.cont

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
        p_WCS[2] = -3.8
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
        pos = self.W_T_Ship[:3, 3] # get current pos
        self.ship.setPos(pos[0], pos[1], pos[2])
        # get heading
        roll, pitch, yaw = extract_RPY(self.W_T_Ship[:3,:3])
        self.ship.setHpr(-90+np.rad2deg(yaw), 0, 0)

        # adjust origin of point light to be over ship
        self.plnp.setPos(pos[0], pos[1], 100)

    def renderShipPath(self, dashed=False):
        # render ship path
        if len(self.ship_path) > 1 and self.counter % 30 == 0:
            self.lines.move_to(Vec3(*self.ship_path[-2]))
            self.lines.draw_to(Vec3(*self.ship_path[-1]))

            line_node = self.lines.create()
            line_node_path = NodePath(line_node)
            line_node_path.setShaderOff()
            line_node_path.reparent_to(self.render)  # Attach to the render

    def purgeBuoyRender(self):
        for light in self.buoy_lights:
            self.render.clearLight(light)
            light.removeNode()
            self.buoy_lights.remove(light)
        for buoy in self.gt_buoys_render:
            buoy.removeNode()
            self.gt_buoys_render.remove(buoy)

        for buoy in self.pred_buoys_render:
            buoy.removeNode()
            self.pred_buoys_render.remove(buoy)

    def renderBuoyGT(self):
        for x, y in self.buoysGT:
            buoy = deepcopy(self.buoy_model_green)  # Load a plane model
            buoy.setScale(15, 15, 15)
            buoy.setPos(x, y, 0)  # Position it at the origin
            p = 3 * np.sin(1.5*self.t)
            r = 1.5 * np.sin(1*self.t)
            buoy.setHpr(0, p, r)
            buoy.setShaderAuto()
            buoy.reparentTo(self.render)  # Add to the scene
            self.gt_buoys_render.append(buoy)
            plight = PointLight("plight")
            plight.setColor((0, 1, 0, 1))
            plight.attenuation = (0.5, 0, 0.0005)
            plnp = buoy.attachNewNode(plight)
            self.buoy_lights.append(plnp)
            self.render.setLight(plnp)

    def renderBuoyPreds(self):
        # the predictions are expected to be in lat lon coords
        for x, y in self.preds:
            buoy = deepcopy(self.buoy_model_red)  # Load a plane model
            buoy.setScale(15, 15, 15)
            p = 3 * np.sin(1.5*self.t+1.5)
            r = 1.5 * np.sin(1*self.t+1.5)
            buoy.setHpr(0, p, r)
            buoy.setPos(x, y, 0)  # Position it at the origin
            buoy.reparentTo(self.render)  # Add to the scene
            self.pred_buoys_render.append(buoy)

            plight = PointLight("plight")
            plight.setColor((1, 0, 0, 1))
            plight.attenuation = (0.5, 0, 0.0005)
            plnp = buoy.attachNewNode(plight)
            self.buoy_lights.append(plnp)
            self.render.setLight(plnp)

    def renderSurface(self):
        self.floor  = self.loader.loadModel("/home/marten/Uni/Semester_4/src/BuoyAssociation/utility/floor5.egg")  # Load a plane model
        self.floor.setScale(100, 100, 1)  # Scale the plane to make it larger
        self.floor.setPos(0, 0, -4)  # Position it at the origin
        self.floor.setShaderAuto()  # Ensure correct shading
        self.floor.reparentTo(self.render)  # Add to the scene

    def setSurfacePos(self, pos):
        self.floor.setPos(*pos[0:2], -4)

    def moveShip(self):
        new_heading = np.radians(np.sin(0.3*self.t) * 45) + self.ship_initHeading
        pos_new_WCS= self.W_T_Ship @ np.array([20*self.dt, 0, 0, 1])
        pos_new_WCS[2] = -3.8
        self.W_T_Ship = T_W_Ship(pos_new_WCS[:3], new_heading)
        if self.counter % 15 == 0:
            self.ship_path.append(pos_new_WCS[0:3])

    def updateLogic(self):
        # logic function, called every 30 ms
        self.t += self.dt
        self.counter += 1
        self.moveShip()

    def Camera(self):
        # cam at World CS pos & heading of 0
        self.camera.setPos(-500, 0, 200)  # X, Y, Z position
        self.camera.lookAt(0, 0, 0)  # Look at the origin (0, 0, 0)
        #gluLookAt(-500, 0, 200, 10, 0, -3, 0, 0, 1) #pos_xyz, view_vector xyz, up axis

    def shipCam(self):
        # cam that follows ship
        pos = np.array([-500, 0, 200, 1])  # camera pos in ship cs
        view_vec = np.array([500, 0, 0, 1])
        pos = self.W_T_Ship@pos
        view_vec = self.W_T_Ship@view_vec

        self.camera.setPos(*pos[:3])  # X, Y, Z position
        self.camera.lookAt(*view_vec[:3])  # Look at the origin (0, 0, 0)

        # adjust plane center to be at location of ship
        self.setSurfacePos(pos[:3])

        #gluLookAt(*pos[:3], *view_vec[:3], 0, 0, 1) #pos_xyz, view_vector xyz, up axis

    def iterate(self):
        if self.follow_ship == False:
            self.Camera()
        else:
            self.shipCam()

    def showScreen(self):
        self.iterate()
        self.purgeBuoyRender()
        self.renderShipIcon()
        self.renderShipPath()
        self.renderBuoyPreds()
        self.renderBuoyGT()

if __name__ == "__main__":
    app = SimpleScene()
    app.run()
