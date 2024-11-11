from OpenGL.GL import *
import OpenGL.GL.shaders
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

        # load shaders
        #vertex_source = self.loadShaders("fog_vertex_shader.vert")
        #fragment_source = self.loadShaders("fog_fragment_shader.frag")

        # # Compile the shader program
        # shader_program = self.compile_shader(vertex_source, fragment_source)
        # glUseProgram(shader_program)

        # # Set fog parameters
        # fog_color = (0.5, 0.6, 0.7)  # Example fog color (light blue)
        # fog_start = 300.0             # Distance at which fog starts
        # fog_end = 1500.0              # Distance at which fog is fully opaque
        # glUniform3fv(glGetUniformLocation(shader_program, "fogColor"), 1, fog_color)
        # glUniform1f(glGetUniformLocation(shader_program, "fogStart"), fog_start)
        # glUniform1f(glGetUniformLocation(shader_program, "fogEnd"), fog_end)


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

    def loadShaders(self, filename):
        with open(filename, 'r') as file:
            return file.read()

    def compile_shader(self, vertex_source, fragment_source):
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, vertex_source)
        glCompileShader(vertex_shader)
        if glGetShaderiv(vertex_shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(vertex_shader).decode('utf-8'))
        
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, fragment_source)
        glCompileShader(fragment_shader)
        if glGetShaderiv(fragment_shader, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(fragment_shader).decode('utf-8'))
        
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)
        if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(shader_program).decode('utf-8'))
        
        # Clean up shaders as they are now linked into the program
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        
        return shader_program


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
            ## draw line from ship to buoy
            p_ship = self.W_T_Ship[:3, 3]
            glColor3f(0.5, 0.5, 0.5)  # Set color to red
            glBegin(GL_LINES)
            a = np.array([x, y, 0]) - p_ship
            points = np.asarray([a*x/(np.linalg.norm(a)/10)+p_ship for x in range(0, int(np.linalg.norm(a)/10))])
            for point in points:
                glVertex3f(point[0], point[1], 0)
            glEnd()
            glPushMatrix()
            # render buoy
            glTranslatef(x, y, 0.0) 
            glColor3f(0.0, 0.8, 0.0)  # Set color to red
            glutSolidSphere(10, 50, 50)  # Draw a sphere with radius, 50 slices, and 50 stacks
            glPopMatrix()

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


# import pygame
# from OpenGL.GL import *
# from OpenGL.GL.shaders import compileProgram, compileShader
# import numpy as np
# from math import sin, cos, tan, radians
# import glm

# # Vertex Shader
# vertex_shader = """
# #version 330
# in vec3 position;
# in vec3 color;
# out vec3 newColor;
# out float dist;
# uniform mat4 model;
# uniform mat4 view;
# uniform mat4 projection;

# void main()
# {
#     gl_Position = projection * view * model * vec4(position, 1.0);
#     float dist = length(view * model * vec4(position,1.0));
#     newColor = color;
# }
# """

# # Fragment Shader
# fragment_shader = """
# #version 330
# in vec3 newColor;
# in float dist;
# out vec4 outColor;

# void main()
# {
#     float fog_maxdist = 8.0;
#     float fog_mindist = 0.1;
#     vec4  fog_colour = vec4(0.5, 0.5, 0.5, 1.0);

#     // Calculate fog
#     float fog_factor = (fog_maxdist - dist) /
#                     (fog_maxdist - fog_mindist);
#     fog_factor = clamp(fog_factor, 0.0, 1.0);

#     outColor = mix(fog_colour, vec4(newColor,1.0), fog_factor);
# }
# """

# # Cube vertices and colors
# vertices = np.array([
#     # Vertices        Colors
#     [-.1, -.1, -.1], [1, 0, 0],
#     [.1, -.1, -.1], [0, 1, 0],
#     [.1, .1, -.1], [0, 0, 1],
#     [-.1, .1, -.1], [1, 1, 0],
#     [-.1, -.1, .1], [0, 1, 1],
#     [.1, -.1, .1], [1, 1, 1],
#     [.1, .1, .1], [0.5, 0.5, 0.5],
#     [-.1, .1, .1], [0.5, 0.5, 0.5],
# ], dtype=np.float32)

# indices = np.array([
#     0, 1, 2, 2, 3, 0,  # Front face
#     1, 5, 6, 6, 2, 1,  # Right face
#     5, 4, 7, 7, 6, 5,  # Back face
#     4, 0, 3, 3, 7, 4,  # Left face
#     3, 2, 6, 6, 7, 3,  # Top face
#     4, 5, 1, 1, 0, 4,  # Bottom face
# ], dtype=np.uint32)

# def init():
#     # Create Vertex Array Object (VAO)
#     vao = glGenVertexArrays(1)
#     glBindVertexArray(vao)

#     # Create Vertex Buffer Object (VBO) and Element Buffer Object (EBO)
#     vbo = glGenBuffers(1)
#     glBindBuffer(GL_ARRAY_BUFFER, vbo)
#     glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

#     ebo = glGenBuffers(1)
#     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
#     glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

#     # Set attribute pointers
#     stride = 6 * vertices.itemsize
#     position_offset = ctypes.c_void_p(0)
#     color_offset = ctypes.c_void_p(3 * vertices.itemsize)
#     glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, stride, position_offset)
#     glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, stride, color_offset)
#     glEnableVertexAttribArray(0)
#     glEnableVertexAttribArray(1)

#     return vao

# def main():
#     # Initialize Pygame
#     pygame.init()
#     width, height = 800, 600
#     pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
#     pygame.display.set_caption("Shader Testing")
#     glViewport(0, 0, width, height)

#     # Compile shaders and create shader program
#     shader_program = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER),
#                                     compileShader(fragment_shader, GL_FRAGMENT_SHADER))
#     model_loc = glGetUniformLocation(shader_program, "model")
#     view_loc = glGetUniformLocation(shader_program, "view")
#     projection_loc = glGetUniformLocation(shader_program, "projection")
#     vao = init()

#     # # Light parameters
#     # light_position_loc = glGetUniformLocation(shader_program, "lightPosition")
#     # light_color_loc = glGetUniformLocation(shader_program, "lightColor")
#     # object_color_loc = glGetUniformLocation(shader_program, "objectColor")

#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         glClearColor(0.0, 0.0, 0.0, 0.0)
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glEnable(GL_DEPTH_TEST)

#         # Define the view and projection matrices (unchanged)
#         W_T_C = np.array([[1.0, 0.0, 0.0, 0],
#                 [0.0, 1.0, 0.0, 0.0],
#                 [0.0, 0.0, 1.0, -3.0],
#                 [0.0, 0.0, 0.0, 1.0]])
#         view = np.linalg.inv(W_T_C)
#         eye = np.array([-3.0, 0.0, 1.0])  # Camera position (looking from 3 units on the Z-axis)
#         center = np.array([0.0, 0.0, 0.0])  # Point the camera is looking at (origin)
#         up = np.array([0.0, 0.0, 1.0])  # Up vector (typically (0, 1, 0) for the Y-axis)

#         # Compute the view matrix using glm.lookAt
#         view = np.array(glm.lookAt(eye, center, up))

#         aspect_ratio = width / height
#         near = 0.1
#         far = 100.0
#         fov = 45.0
        
#         projection = np.array(glm.perspective(glm.radians(fov), aspect_ratio, near, far))

#         light_position = np.array([1.0, -1.0, 1.0], dtype=np.float32)
#         light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White light
#         object_color = np.array([1.0, 0.5, 0.5], dtype=np.float32)  # Red object

#         # Define the model matrix (rotate the cube)
#         angle = pygame.time.get_ticks() / 1000.0  # Rotate the cube over time
#         model = np.array([[cos(angle), -sin(angle), 0, 20],
#                  [sin(angle), cos(angle), 0, 0],
#                  [0, 0, 1, 0],
#                  [0, 0, 0, 1]])

#         model2 = np.array([[cos(angle), -sin(angle), 0, 0],
#                  [sin(angle), cos(angle), 0, 1],
#                  [0, 0, 1, 0],
#                  [0, 0, 0, 1]])
#         model3 = np.array([[cos(angle), -sin(angle), 0, 0],
#                  [sin(angle), cos(angle), 0, -1],
#                  [0, 0, 1, 0],
#                  [0, 0, 0, 1]])

#         # Use the shader program
#         glUseProgram(shader_program)

#         # Set the model, view, and projection matrices in the shader 
#         glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
#         glUniformMatrix4fv(projection_loc, 1, GL_TRUE, projection)

#         glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)

#         # glUniform3fv(light_position_loc, 1, light_position)
#         # glUniform3fv(light_color_loc, 1, light_color)
#         # glUniform3fv(object_color_loc, 1, object_color)

#         glBindVertexArray(vao)
#         glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)


#         glUniformMatrix4fv(model_loc, 1, GL_TRUE, model2)
#         glBindVertexArray(vao)
#         glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

#         glUniformMatrix4fv(model_loc, 1, GL_TRUE, model3)
#         glBindVertexArray(vao)
#         glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

#         pygame.display.flip()

#     pygame.quit()


# if __name__ == "__main__":
#     main()