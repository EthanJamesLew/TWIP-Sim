from pid import IterPID
from motor import PWMDCMotor
from twip import TWIPZi
from system import SysBase

from PyQt5 import QtGui, QtWidgets, QtOpenGL, QtCore
from twip_widget import TWIPWidget

import numpy as np

class PIDRobot(SysBase):
    def __init__(self, Ts):
        self.sp_tilt = 0
        self.sp_yaw = 0

        self.twip = TWIPZi()

        self.motor_l = PWMDCMotor()
        self.motor_r = PWMDCMotor()
        self.motor_l.set_IC([0, 0])
        self.motor_r.set_IC([0, 0])

        self.pid_tilt = IterPID(Ts)
        self.pid_yaw = IterPID(Ts)

        self.pid_tilt.tune(70, 5, 100)
        self.pid_yaw.tune(25, 2, 0)
        
        self.pid_tilt.set_IC([0, 0, 0, 0])
        self.pid_yaw.set_IC([0, 0, 0, 0])

        self.parameters = self.twip.parameters

    def update_current_state(self, dt, F =None):
        # get twip state
        coords = self.twip.get_position_coordinates()
        curr_yaw = coords[2]
        curr_tilt = coords[5]
        
        #print(coords)
        # Get error signals
        err_y = self.sp_yaw - curr_yaw
        err_t = self.sp_tilt - curr_tilt
        #print(err_t)

        # Update PID
        self.pid_yaw.update_current_state(dt, [err_y])
        self.pid_tilt.update_current_state(dt, [err_t])

        # Get PID values
        ctrl_t = self.pid_tilt.get_position_coordinates()
        ctrl_y = self.pid_yaw.get_position_coordinates()

        # Convert to PWM values
        pwm_t = int(ctrl_t)
        pwm_y = int(ctrl_y)

        #print(pwm_t, pwm_y)

        # Update Motors
        self.motor_l.update_current_state(dt, [-pwm_t + pwm_y])
        self.motor_r.update_current_state(dt, [-pwm_t - pwm_y])

        

        # Get motor torques
        t_l = self.motor_l.get_position_coordinates()[0]
        t_r = self.motor_r.get_position_coordinates()[0]

        tF = [t_l*10, t_r*10,  0, 0]


        if F is not None:
            tF = [(tF[i] + F[i]) for i in range(0, len(F))] 

        # Update TWIP
        self.twip.update_current_state(dt, tF)

    def set_tilt(self, tilt):
        self.sp_tilt = tilt

    def set_yaw(self, yaw):
        self.sp_yaw = yaw

    def set_IC(self, coord):
        self.twip.set_IC(coord)

    def get_position_coordinates(self):
        return self.twip.get_position_coordinates()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    '''
    twip = PIDRobot(0.01)

    # Simulation Parameters
    nsteps = 200
    dt = 1/100.
    t = np.zeros(nsteps)

    pos = np.zeros((nsteps, 4))
    twip.set_IC([0.0, 0, 0.04, 0, 0.0, 0])

    # Fake impulse
    #twip.update_current_state(dt, [1/dt*0.1, 1/dt*0.5,  .2, 0]) 

    
    for i in range(0, nsteps):
        cpos = twip.get_position_coordinates()
        pos[i, 0] = cpos[2]
        pos[i, 1] = cpos[5]
        pos[i, 2] = twip.motor_l.get_position_coordinates()[0]

        t[i] = i*dt
        twip.update_current_state(dt, [0, 0,  0, 0])

    plt.plot(t, pos[:, :])
    plt.show()
    '''
    class MainWindow(QtWidgets.QMainWindow):
        ''' Realtime TWIP viewer program
        '''
        def __init__(self):
            super(MainWindow, self).__init__()

            # Create TWIP model
            self.twip = PIDRobot(0.03)
            self.twip_widget = TWIPWidget(self, self.twip)

            # Add layout to put twip_widget in
            wid = QtWidgets.QWidget(self)
            self.setCentralWidget(wid)
            mainLayout = QtWidgets.QHBoxLayout()
            mainLayout.addWidget(self.twip_widget)
            wid.setLayout(mainLayout)

            # Setup twip initial state
            dt = 1/30
            self.twip.set_IC([0, 0, 0, 0, 0, 0])
            self.twip.update_current_state(dt, [1/dt*0.01, -1/dt*0.009,  0, 0]) 
            self.dt = dt
            
        def update_twip(self):
            ''' program mainloop method
            '''
            self.twip.update_current_state(self.dt, [0, 0,  0, 0])
            self.twip_widget.draw_twip()

    app = QtWidgets.QApplication(['TWIP Viewer'])
    window = MainWindow()
    window.resize(200, 200)
    
    sim_timer = QtCore.QTimer()
    sim_timer.timeout.connect(window.update_twip)
    sim_timer.start(1/90*1000)

    window.show()

    app.exec_()


    
