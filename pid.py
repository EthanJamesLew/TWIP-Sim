''' PID System Controller

Ethan Lew
4/18/19
elew@pdx.edu

Create a PID controller capable of being used with SysBase Dynamics object 
'''

from system import IterSysBase, wraptopi, minmax
import numpy as np
from numpy import sin, cos
from numba import jit

import enum 
class IntegrationType(enum.Enum): 
    rectangular = 0
    trapezoidal = 1
    parabolic = 2
  

class PIDIntegralError(Exception):
   pass

class IterPID(IterSysBase):
    '''Discrete Time PID Controller

    Kp - proportional gain
    Kd - derivative gain
    Ki - integral gain

    Types:
    'linear' - linear control input
    'angular' - angular control input (wraps to [-pi, pi])

    Integral Methods:
    'Rectangular' - accumulator
    'Trapezoidal' - Newton's Integration Method
    'Parabolic' - Quadratic Simpson's Method

    Derivative Methods:
    'backwards' - Backwards difference

    TODO: Implement kalman filter as a denoising option
    '''
    def __init__(self, Ts, Tp=0.01):
        IterSysBase.__init__(self, Ts, Tp=Tp, n=1)
        default_PID = {'Kp': 1.0, 'Kd': 1.0, 'Ki': 1.0,
                         'max': 10000.0, 'i_max': 10000.0, 'd_max': 10000.0, 'd_off': 0,
                         'i_method' : IntegrationType.trapezoidal, 'd_method' : 'backwards', 'type' : 'linear'}
        self.parameters = default_PID
        self.equations = 'PID'

        self.q = np.zeros((4, 1))
        self.p = np.zeros((1, 1))

        self.force = np.zeros((1))

        self.update_params()

    def tune(self, Kp, Kd, Ki):
        self.parameters['Kp'] = Kp
        self.parameters['Kd'] = Kd
        self.parameters['Ki'] = Ki

    def update_params(self):
        method = self.parameters['i_method']


    def vdq(self, t, q, F):
        method = self.parameters['i_method']
        pid_type = self.parameters['type']

        # If angular, unwrap phase to continuous phase
        if(pid_type == 'angular'):
            F = [wraptopi(F[0])]

        dq = np.zeros((4))
        dq[0] = F[0]
        dq[1] = q[0] 
        dq[2] = q[1]

        # Apply integration method
        if(method == IntegrationType.rectangular):
            dq[3] = q[3] + (F[0]/2 + dq[1]/2)*self.Ts
        elif(method == IntegrationType.trapezoidal):
            dq[3] = q[3] + ((F[0] + 4*dq[1] + dq[2])/6)*self.Ts
        elif(method == IntegrationType.parabolic):
            dq[3] = q[3] + F[0]*self.Ts
        else:
            raise PIDIntegralError
        
        return dq

    def convert_sys(self):
        pid = self.parameters
        Kp = pid['Kp']
        Ki = pid['Ki']
        Kd = pid['Kd']

        i_max = pid['i_max']
        d_max = pid['d_max']
        tot_max = pid['max']
        d_off = pid['d_off']
        
        dp = np.zeros((1))
        p_term = Kp*self.q[0]
        i_term = minmax(Ki*(self.q[3]), i_max)
        
        # Explore denoising options
        d_term = minmax(Kd*(self.q[0] - self.q[1])/self.Ts, d_max)

        dp[0] = minmax(p_term + i_term + d_term, tot_max)

        # Add d offset
        if dp[0] < 0:
            dp[0] -= d_off
        else:
            dp[0] += d_off

        self.p = dp

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PID = IterPID(1)

    print(PID)
    nsteps = 340
    dt = 1/10
    t = np.zeros(nsteps)

    pos = np.zeros((nsteps, 3))

    PID.set_IC([0, 0, 0, 0])

    for i in range(0, nsteps):
        pos[i, 0] = i*dt
        pos[i, 1] = PID.get_position_coordinates()
        t[i] = i*dt
        PID.update_current_state(dt, [(i*dt)])
        pos[i, 2] = (i*dt)

    plt.plot(pos[:, 0], pos[:, 1:])
    plt.show()