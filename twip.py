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
from system import SysBase, wraptopi, rk4

class TWIPZi(SysBase):
    '''TWIP System described in Z. Li et al, Advanced Control of Wheeled Inverted Pendulums, Springer-Verlag London 2013
    
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
    '''
    def __init__(self):
        SysBase.__init__(self, n=6)
        self.equations = "Zi et al."
        self.q = np.zeros((6))
        self.p = np.zeros((5))
        self.kinematic_coordinates = np.zeros((5, 1))
        self.force = np.zeros((0))

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

    plt.plot(t, pos[:, 1])
    plt.show()