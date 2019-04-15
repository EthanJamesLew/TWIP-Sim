'''DC Motor Model

Ethan Lew
4/14/2019
elew@pdx.edu


'''

import numpy as np
from numpy import sqrt
from system import SysBase

class DCMotor(SysBase):
    ''' DC Motor Description

    The DC motor description incoporates the motor parameters:

        J (kg m^2) - Rotational inertia of the load attached to the armature
        L (H) - Motor Inductance
        R (Ohms) - Motor Resistance
        Km (V s) - proportionality constant tau = Km * I
        Ke (V s / rad) - proportionality constant EMF = Ke * omega
    '''
    def __init__ (self):
        SysBase.__init__(self, n=2)
        # these values were taken from a pololu motor: https://www.pololu.com/product/3202, 550 RPM model
        default_motor = {'J' : 127e-6, 'L': 15.38e-3, 'R': 11.1, 'Km': 0.0993, 'Ke': 0.1893}
        self.parameters = default_motor
        self.equations = 'Vukosavic'

        # holds [current, angular rate]
        self.q = np.zeros((2, 1))
        # holds [torque]
        self.p = np.zeros((1, 1))
        self.kinematic_coordinates = np.zeros((1, 1))

        # holds [Vrms]
        self.force = np.zeros((1))

    def vdq(self, t, q, F):
        mtr = self.parameters
        R = mtr['R']
        L = mtr['L']
        Ke = mtr['Ke']
        Km = mtr['Km']
        J = mtr['J']

        dq = np.zeros((2))
        dq[0] = (-R/L)*q[0] - (Ke/L)*q[1] + F[0]
        dq[1] =  (Km/J)*q[0] + (0)*q[1]
        return dq

    def convert_sys(self):
        self.p = self.parameters['Km'] * self.q 

    def vdp(self, t, q, ic):
        return np.zeros((2))

class PWMDCMotor(DCMotor):
    ''' PWM DC Motor

    This is a very simple converter that accepts 8 bit duty cycle numbers and 
    converts it into Vrms for the DCMotor model. No new dynamics are introduced.
    '''
    def __init__(self):
        DCMotor.__init__(self)
        self.parameters['resolution'] = 8
        self.parameters['Vs'] = 12

    def convert_pwm(self, F):
        Vs = self.parameters['Vs']
        res = self.parameters['resolution']
        pwm = min(max(F[0], 0), 2**res-1)
        return np.array([Vs*sqrt(pwm/res)])

    def vdq(self, t, q, F):
        return super(PWMDCMotor, self).vdq( t, q, self.convert_pwm(F))

    def set_force(self, F):
        self.force = self.convert_pwm(F)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    motor = PWMDCMotor()

    print(motor)

    nsteps = 340
    dt = 1/10.
    t = np.zeros(nsteps)

    pos = np.zeros((nsteps, 2))
    motor.set_IC([0, 0])

    for i in range(0, nsteps):
        pos[i, 0] = i*dt
        pos[i, 1] = motor.get_position_coordinates()[0]
        t[i] = i*dt
        motor.update_current_state(dt, [255])
        

    plt.plot(pos[:, 0], pos[:, 1])
    plt.show()



    




