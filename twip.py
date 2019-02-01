'''TWIP Model

Ethan Lew
1/26/2019
elew@pdx.edu 

Naming Conventions:

Differentials: d<variable> (e.g. dt, dp)
Differential Systems: vd<coord> (e.g. vdq)  
Coordinates: q is preferred for generalized DYNAMICS coordinates, p is preferred for POSITIONAL coordinates
Current States: c<name> (e.g. ct)

Style Conventions: 

In general, prefer snake case over camel case (except for class names). Uphold the standards outlined by PEP8 when
it is logical to do so.  Math functionality should use be short and ideally one word. Public interface functions 
should have the proper accessor conventions and be descriptive in general.

Comments in implementation should be formatted as

# <quick comment>
code

Comments over classes should be formatted as

class className(object):
'' '<Verbose Description>
'' '

'''

import numpy as np
from numpy import sin, cos

'''
TWIP Custom Error 
'''
class SysNoParameterError(Exception):
   pass

def rk4(f):
    '''Generic implementation of the Runge-Kutta solver with Dormand-Prince weights

    Args:
        f: A function f(t, y), such that y' = f(t, y)
    Returns:
        y solution 

    This system doesn't have any error free guarantees that can be found in scipy's integrate.solve_ivp() method. In 
    general, scipy's solution isn't ideal for the simulation approach used here. Adaptive time methods may be worth
    exploring in the future.
    '''
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
                        )( dt * f( t + dt  , y + dy3   ) )
	                    )( dt * f( t + dt/2, y + dy2/2 ) )
	                    )( dt * f( t + dt/2, y + dy1/2 ) )
	                    )( dt * f( t       , y         ) )

class SysBase():
    '''Holds the interface, parameters and initial conditions for the simulation of a system. 
    
    The parameter schema is not fixed as different algorithms require different parameters for their outcomes. 
    For rendering to work, the following values should be defined somewhere:

        l - distance from the track to the COG
        d - length of the track
        r - radius of the wheel

    The TWIP system can have a variety of generalized coordinate descriptions. Only one interpretation can be used by the
    viewer, though, so a method is available to convert to this form (get_position_coordinates)

        (x, y) - position of the TWIP
        theta  - TWIP yaw angle
        alpha  - TWIP tilt angle
        thetar - right wheel angle
        thetal - left wheel angle

    Dynamics coordinates need to be held in self.q and kinematic (positional) coordinates are held in self.p

    Implementation names (authors preferred) are held in a field called self.equations. Cite responsibly.

    Needs: SysBase describes the state of a system over time. To use properly, a timing system is needed. Also, a
    controller class should be responsible for selecting the F values to achieve stability. Finally, an optional sensor 
    class can model the delay and noise present in a realized TWIP system.
    '''
    def __init__(self, n = 3):
        default_bot = {"Mw": 0.8, "Iw": 0.02, "r" : .2, "m" : 0.5,
             "l" : .15, "d" : .6, "M" : 0.7, "IM": 0.08,
              "Ip": 0.06, "g": 9.81 }

        self.parameters = {}
        self.equations = "None"
        self.ct = 0
        self.parameters = default_bot


    def set_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameter(self, name):
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise SysNoParameterError

    def set_IC(self, coord):
        self.q = coord 

    def get_IC(self):
        return self.q

    def update_current_state(self, dt, F):
        if F is None:
            F = np.zeros((6))
        # Update dynamics
        self.dq = rk4(lambda t, q: self.vdq(t, q, F))
        self.ct, self.q = self.ct + dt,  self.q + self.dq( self.ct, self.q, dt )
        
        self.convert_sys()

        # Update kinematics
        self.dp = rk4(lambda t, q: self.vdp(t, q, self.q))
        self.p = self.p + self.dp( self.ct, self.p, dt )

    def get_position_coordinates(self):
        raise NotImplementedError
    
    def vdp(self, t, q, ic):
        raise NotImplementedError

    def vdq(self, t, q, F):
        raise NotImplementedError

    def convert_sys(self):
        raise NotImplementedError

    def reset_time(self):
        self.ct = 0

class TWIPZi(SysBase):
    '''TWIP System described in Z. Li et al, Advanced Control of Wheeled Inverted Pendulums, Springer-Verlag London 2013
    '''
    def __init__(self):
        SysBase.__init__(self, n=6)
        self.equations = "Zi et al."
        self.q = np.zeros((6))
        self.p = np.zeros((5))
        self.kinematic_coordinates = np.zeros((5, 1))

    def get_position_coordinates(self):
        return np.concatenate((self.p, [self.q[2]]))

    def vdp(self, t, q, ic):
        theta = self.q[1]
        v = self.q[3]
        omega = self.q[4]
        rbt = self.parameters
        r =    rbt['r']
        d =     rbt['d']
        
        qp = np.zeros((5))
        qp[0] = cos(theta)*v
        qp[1] = sin(theta)*v 
        qp[2] = omega
        qp[3] = v/r + omega/d 
        qp[4] = v/r - omega/d 
        return qp

    def vdq(self, t, q, F):
        # Unpack Parameters
        rbt = self.parameters
        M =     rbt['M']
        Mw =    rbt['Mw']
        m =     rbt['m']
        Iw =    rbt['Iw']
        Ip =    rbt['Ip']
        Imm =    rbt['IM']
        r =    rbt['r']
        l =     rbt['l']
        d =     rbt['d']
        g = rbt['g']

        # Unpack forcing parameters
        tl =    F[0]
        tr  =    F[1]
        dl =    F[2]
        dr  =   F[3]
        
        # Calculate differential
        qp = np.zeros((6))
        qp[0] = q[3]
        qp[1] = q[4]
        qp[2] = q[5]
        qp[3] =((m*l**2+Imm)*(-m*l*q[5]**2*sin(q[2])-tl/r-tr/r-dl-dr)+m**2*l**2*cos(q[2])*g*sin(q[2]))/((m*l**2+Imm)*(M+2*Mw+m+2*Iw/r**2)-m**2*l**2*cos(q[2])**2)
        qp[4] = 2*d*(tl/r-tr/r+dl-dr)/(Ip+(2*(Mw+Iw/r**2))*d**2)
        qp[5] =  (m*l*cos(q[2])*(-m*l*q[5]**2*sin(q[2])-tl/r-tr/r-dl-dr)+m*g*l*sin(q[2])*(M+2*Mw+m+2*Iw/r**2))/((m*l**2+Imm)*(M+2*Mw+m+2*Iw/r**2)-m**2*l**2*cos(q[2])**2) 
                
        return qp

    def convert_sys(self):
        self.p[2] = self.q[1]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    twip = TWIPZi()

    # Simulation Parameters
    nsteps = 340
    dt = 1/60.
    t = np.zeros(nsteps)

    pos = np.zeros((nsteps, 2))
    twip.set_IC([0, 3.14/4, 0, 0, 0, 0])

    # Fake impulse
    twip.update_current_state(dt, [1/dt*0.1, 1/dt*0.5,  .2, 0]) 

    for i in range(0, nsteps):
        cpos = twip.get_position_coordinates()
        pos[i, 0] = cpos[0]
        pos[i, 1] = cpos[1]
        t[i] = i*dt
        twip.update_current_state(dt, [0, 0,  0, 0])

    plt.plot(pos[:, 0], pos[:, 1])
    plt.show()