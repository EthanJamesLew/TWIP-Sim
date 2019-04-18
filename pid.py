from system import IterSysBase
import numpy as np
from numpy import sin

def minmax(x, s):
    return min(max(x, -s), s)

class IterPID(IterSysBase):
    '''
    '''
    def __init__(self, Ts, Tp=0.01):
        IterSysBase.__init__(self, Ts, Tp=Tp, n=1)
        default_PID = {'Kp': 1.0, 'Kd': 1.0, 'Ki': 1.0, 'max': 1000.0, 'i_max': 1000.0, 'd_max': 1000.0}
        self.parameters = default_PID
        self.equations = 'PID'

        self.q = np.zeros((4, 1))
        self.p = np.zeros((1, 1))

        self.force = np.zeros((1))


    def vdq(self, t, q, F):
        dq = np.zeros((4))
        dq[0] = F[0]
        dq[1] = q[0] 
        dq[2] = q[1]
        dq[3] = q[3] + F[0]
        return dq

    def convert_sys(self):
        pid = self.parameters
        Kp = pid['Kp']
        Ki = pid['Ki']
        Kd = pid['Kd']

        i_max = pid['i_max']
        d_max = pid['d_max']
        tot_max = pid['max']
        
        dp = np.zeros((1))
        p_term = Kp*self.q[0]
        i_term = minmax(Ki*(self.q[3])*self.Ts, i_max)
        d_term = minmax(Kd*(self.q[0] - self.q[1])/self.Ts, d_max)

        
        dp[0] = minmax(p_term + i_term + d_term, tot_max)
        self.p = dp

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PID = IterPID(.01)

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
        PID.update_current_state(dt, [sin(i*dt)])
        pos[i, 2] = sin(i*dt)

    plt.plot(pos[:, 0], pos[:, 1:])
    plt.show()