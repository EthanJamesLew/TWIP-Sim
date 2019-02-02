from twip import TWIPZi
from twip_widget import TWIPWidget, wraptopi
import numpy as np
from numpy import sign

class PIDController(object):
    def __init__(self, twip, param, kp = 0, kd = 0, ki = 0):
        self.set = 0
        self.twip = twip
        self.param = param
        self.lstate = self.set-wraptopi(twip.get_position_coordinates()[self.param])
        self.I = np.zeros(self.lstate.shape)
        self.D = np.zeros(self.lstate.shape)
        self.P = kp*self.lstate

        self.kp = kp
        self.kd = kd
        self.ki = ki

        self.sign = False

    def set_set(self,s):
        if (sign(self.set) != sign(s)):
            self.sign = True
        self.set = s

    def update_F(self, dt):
        cstate =  wraptopi( self.set - self.twip.get_position_coordinates()[self.param])
        
        if self.sign:
            self.I = np.zeros(self.lstate.shape)
            self.D = np.zeros(self.lstate.shape)
            self.P = self.kp*self.lstate
            self.sign = False
        else:
            self.I = self.ki*(self.I + dt*cstate)
            self.D = self.kd*(cstate - self.lstate)/dt
            self.P = self.kp*cstate

        total = self.P + self.I + self.D
        total = np.sign(total)*min(abs(total), 5)
        #print('t:', total)
        if self.param == 5 or self.param == 0:
            F = [-total, -total, 0, 0]
        elif self.param == 2:
            F = [total, -total, 0, 0]
        self.lstate = cstate       
        return F

if __name__ == "__main__":
    from PyQt5 import QtGui, QtWidgets, QtOpenGL, QtCore
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    import pygame
    from pygame.locals import *

    class MainWindow(QtWidgets.QMainWindow):
        ''' Realtime TWIP viewer program
        '''
        def __init__(self):
            super(MainWindow, self).__init__()

            QtWidgets.qApp.installEventFilter(self)

            # Create TWIP model
            self.twip = TWIPZi()
            self.theta = np.pi/4 + 0.01 

            self.tilt_pid = PIDController(self.twip, 5, 10, 4, 0)
            self.yaw_pid = PIDController(self.twip, 2, 4, 1, 0)
            self.twip_widget = TWIPWidget(self, self.twip)

            self.yaw_pid.set_set(self.theta)

            # Add layout to put twip_widget in
            wid = QtWidgets.QWidget(self)
            self.setCentralWidget(wid)
            mainLayout = QtWidgets.QHBoxLayout()
            mainLayout.addWidget(self.twip_widget)
            wid.setLayout(mainLayout)

            # Setup twip initial state
            dt = 1/30
            self.twip.set_IC([0, 0, 0, 0, 0, 0])
            self.twip.update_current_state(dt, F = [1/dt*0.5, -1/dt*0.4,  0, 0]) 
            self.dt = dt

            pygame.display.init()
            pygame.joystick.init()
            
            joystick_count = pygame.joystick.get_count()
            for i in range(joystick_count):
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                name = joystick.get_name()

                if "Logitech" in name:
                    print("Registering %s" % name)
                    self.joystick = joystick
                    self.name = name
                    self.axes = joystick.get_numaxes()



        def update_twip(self):
            ''' program mainloop method
            '''
            Ft = self.tilt_pid.update_F(self.dt)
            Fy = self.yaw_pid.update_F(self.dt)
            F = np.array(Ft) + np.array(Fy)

            pygame.event.pump()

            
            #for i in range(self.axes):
            #    axis = self.joystick.get_axis(i)
            #    print(self.name)
            #    print("Axis %d: value %6.3f" % (i, axis))

            #print("\n")
            

            tiltf = self.joystick.get_axis(1)
            self.tilt_pid.set_set(-tiltf/5)

            yawf = self.joystick.get_axis(3)
            self.theta = wraptopi(self.theta - yawf/10)
            self.yaw_pid.set_set(self.theta)
            
            self.twip.update_current_state(self.dt, list(F))
            self.twip_widget.draw_twip()
            

        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.KeyPress:
                if event.key() == QtCore.Qt.Key_A:
                    self.theta = wraptopi(self.theta + np.pi / 8)
                    self.yaw_pid.set_set(self.theta)
                    return 1
                elif event.key() == QtCore.Qt.Key_D:
                    self.theta = wraptopi(self.theta - np.pi / 8)
                    self.yaw_pid.set_set(self.theta)
                    return 1
            return super().eventFilter(obj, event)

    app = QtWidgets.QApplication(['TWIP Viewer'])
    window = MainWindow()
    window.resize(500, 500)
    
    sim_timer = QtCore.QTimer()
    sim_timer.timeout.connect(window.update_twip)
    sim_timer.start(1/90*1000)

    window.show()

    app.exec_() 




