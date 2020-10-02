import pygame as pg
import numpy as np
import random
import math
import time

from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward, predict
from building_your_deep_neural_network_step_by_step_v8a import *

vec = pg.math.Vector2
#####################
#v. 5: lqr implemented
#v. 6: manual disturb
#v. 6_nn_3: implemented controller_nn
#v. 6_nn_4: fixed infinite manual torque issue, implemented runge-kutta, implemented cases for training data creation


############ Some color codes and other constants ############

WHITE = (255, 255, 255)
BLUE = (0,   0, 255)
GREEN = (0, 255,   0)
RED = (255,   0,   0)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)
TEXTCOLOR = (0,   0,  0)
VEL_MAX = 10
###########################################
(width,height)=(800,700)
dotStartPos = (int(width/2), int(height/2))
running = True
testing = True
###########################################
def convert_tuple_int(test_tuple):
    x = vec[int(test_tuple[0]),int(test_tuple[1])]
    return x

def pixel_2_meter(n_pixel):
    n_meter = int(n_pixel*dx)
    return n_meter

def meter_2_pixel(n_meter):
    n_pixel = int(n_meter/dx)
    return n_pixel

###########################################

    
class Pendulum:
    def __init__(self, window, color, init_ang_pos, mass, l):
        
        self.window = window
        self.color = color
        self.mass = mass
        self.l = l
        self.l_pixel = meter_2_pixel(self.l)
        self.ang_pos = init_ang_pos
        self.ang_vel = 0

    def runge_kutta(self, pos, vel, acc):
        #recycled from http://doswa.com/2009/01/02/fourth-order-runge-kutta-numerical-integration.html

        x = pos
        v = vel
        a = acc
        
        x1 = x
        v1 = v
        a1 = a

        x2 = x + 0.5*v1*dt
        v2 = v + 0.5*a1*dt
        a2 = -g*math.cos(x2)/self.l

        x3 = x + 0.5*v2*dt
        v3 = v + 0.5*a2*dt
        a3 = -g*math.cos(x3)/self.l

        x4 = x + v3*dt
        v4 = v + a3*dt
        a4 = -g*math.cos(x4)/self.l

        pos_f = x + (dt/6.0)*(v1 + 2*v2 + 2*v3 + v4)
        vel_f = v + (dt/6.0)*(a1 + 2*a2 + 2*a3 + a4)
        
        return pos_f, vel_f
 
    def move(self, torque):
        self.ang_acc = -g*math.cos(self.ang_pos)/self.l + torque/(self.mass * self.l**2)        
##        self.ang_vel = self.ang_vel + self.ang_acc*dt
##        self.ang_pos = self.ang_pos + self.ang_vel*dt
        
        self.ang_pos, self.ang_vel = self.runge_kutta(self.ang_pos, self.ang_vel, self.ang_acc)
        self.pos = (int(width/2) + int(self.l_pixel*math.cos(self.ang_pos)), - int(height/2) + int(height - self.l_pixel*math.sin(self.ang_pos)))

        while self.ang_pos >= 2*math.pi:
            self.ang_pos = self.ang_pos - 2*math.pi
        while self.ang_pos <= -2*math.pi:
            self.ang_pos = self.ang_pos + 2*math.pi    
        
        self.show()
        
        
    def show(self):
        pg.draw.circle(self.window, self.color, self.pos, 10)
        pg.draw.line(self.window, WHITE, (int(width/2),int(height/2)), self.pos)
        pg.display.update()

class Controller_manual:
    def __init__(self):
        self.torque = 0

    def update_torque(self):
        if pg.key.get_pressed()[pg.K_LEFT]:
            self.torque += 0.1
        if pg.key.get_pressed()[pg.K_RIGHT]:
            self.torque += -0.1
        for event in pg.event.get():
            if event.type == pg.KEYUP:
                if event.key == pg.K_LEFT or event.key == pg.K_RIGHT:
                   self.torque = 0
        return self.torque

class Controller_automatic:
    def __init__(self):
        self.torque = 0
        self.k =  np.array([[35.2768, 11.6309]])
        self.y = np.zeros((2,1))

        self.training_data = np.array([[], []], ndmin = 2)
        self.training_torques = np.array([])

    def update_torque(self, ang_pos, ang_vel):
        self.y[0][0] = ang_pos - math.pi/2
        self.y[1][0] = ang_vel - 0
        self.ang_pos = ang_pos
        self.ang_vel = ang_vel
        aux_torque = np.matmul(self.k, self.y)
        self.torque = -aux_torque[0]
        
        
        self.build_training_data(self.ang_pos, self.ang_vel, self.torque)
        return self.torque
    
    def build_training_data(self, ang_pos, ang_vel, torque):
        #print(self.training_data.shape)
        #print(str(np.array([[ang_pos],[ang_vel]]).shape) + "seria essa?")
        #print(np.array([[ang_pos],[ang_vel]]))
        #print(ang_pos)
        #print(ang_vel)
        aux = np.array([[ang_pos],[ang_vel]]).reshape(2, 1)
        
        self.training_data = np.append(self.training_data, aux,  axis = 1)
        self.training_torques = np.append(self.training_torques, torque)
        #print(self.training_data.shape)
        #print(self.training_torques.shape)
        return self.training_data, self.training_torques

class Controller_neural_net:
    def __init__(self, seed):
        self.torque = 0

        self.W1 = np.loadtxt('W1.txt', dtype=float).reshape(1, -1)
        self.b1 = np.loadtxt('b1.txt', dtype=float).reshape(1, -1)
        self.W2 = np.loadtxt('W2.txt', dtype=float).reshape(1, -1)
        self.b2 = np.loadtxt('b2.txt', dtype=float).reshape(1, -1)
        
    def update_torque(self, ang_pos, ang_vel):

        X = np.array([[ang_pos],[ang_vel]]).reshape(2, 1)
        X = X/10 #normalization
        
        A1, cache1 = linear_activation_forward(X, self.W1, self.b1, "relu")
        A2, cache2 = linear_forward(A1, self.W2, self.b2)
        
        self.torque = A2
        return self.torque
          

###########################################

# Initiliaze pygame #
def main():
    global g, dt, dx
    

    g = 9.8
    dt = 10/1000
    dx = 1/250

    initial_angle_position = math.pi*(170/180)

    mass = 0.5
    length_meters = 1   
    
    pg.init()
    clock = pg.time.Clock()
    # Make screen and filling it with color
    window = pg.display.set_mode((width, height))
    #pendulum = Pendulum(window, WHITE, initial_angle_position, mass, length_meters)

    controller_nn = Controller_neural_net(-3)
    controller_automatic = Controller_automatic()
    controller_manual = Controller_manual()
    
    state = 'MOVING'

    torque = 0

    clock = pg.time.Clock()
    create_training_data = False
    
    if create_training_data:
        for initial_angle_position in np.linspace(3*math.pi/2 - 0.1,-math.pi/2 + 0.1, 11):
            pendulum = Pendulum(window, WHITE, initial_angle_position, mass, length_meters)
            equilibrium = False
            
            while state == 'MOVING' and equilibrium == False:
                dt = clock.tick()/1000
                #torque = controller_automatic.update_torque(pendulum.ang_pos, pendulum.ang_vel) + controller_manual.update_torque()
                #torque = controller_nn.update_torque(pendulum.ang_pos, pendulum.ang_vel) + controller_manual.update_torque()
                torque = controller_manual.update_torque()
                pendulum.move(torque) 
                pendulum.show()
                window.fill(BLACK)
                
                if pendulum.ang_pos < math.pi/2 + 0.01 and pendulum.ang_pos > math.pi/2 - 0.01:
                    equilibrium = True
    else:
        pendulum = Pendulum(window, WHITE, initial_angle_position, mass, length_meters)
        while state == 'MOVING':
            dt = clock.tick()/1000
            #torque = controller_automatic.update_torque(pendulum.ang_pos, pendulum.ang_vel) + controller_manual.update_torque()
            torque = controller_nn.update_torque(pendulum.ang_pos, pendulum.ang_vel) + controller_manual.update_torque()
            #torque = controller_manual.update_torque()
            pendulum.move(torque) 
            pendulum.show()
            window.fill(BLACK)
            
    np.savetxt('training_data.txt', controller_automatic.training_data, fmt='%f')
    np.savetxt('training_torques.txt', controller_automatic.training_torques.T, fmt='%f')


main()




    


        
    




