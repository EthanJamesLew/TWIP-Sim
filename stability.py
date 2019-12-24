''' Stability Analysis

Ethan Lew
12/12/2019
elew@pdx.edu

Methods to perform stability analysis
'''

import numpy as np
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from multiprocessing import Pool

def is_stable(twip, IC, t_max=10, inner_radius=0.18, outer_radius=1):
    ''' is_stable
    :param twip: TWIPZi object
    :param IC: (4) initial condition for twip 
    :param t_max (10): maximum amount of simulation time (if doesn't settle within t_max, 
                        consider the system unstable) 
    :param inner_radius (0.18): Innermost radius that is considered stable
    :param outer_radius (1): Outermost radius that is considered unstable
    '''
    dt = 1 / 50
    twip.set_IC(IC)
    twip.motor_l.set_IC([0, 0])
    twip.motor_r.set_IC([0, 0])
    twip.pid_tilt.set_IC([0, 0, 0, 0])
    twip.pid_yaw.set_IC([0, 0, 0, 0])

    twip.update_current_state(dt, [0, 0, 0, 0])
    t = 0
    while t < t_max:
        twip.update_current_state(dt, [0, 0, 0, 0])
        t += dt
        cpos = twip.twip.q[1:]
        if np.linalg.norm(cpos) <= inner_radius:
            return True
        elif np.linalg.norm(cpos) >= outer_radius:
            return False
    return False

class IterSample(object):
    ''' IterSample
    IterSample is a way to globally declare a function-like object so that Pool can use it
    This is because Pools require pickling, meaning that lambdas cannot be used.
    '''
    def __init__(self, twip):
        '''
        :param twip: TWIPZi object
        '''
        self.twip = twip
    def __call__(self, X):
        '''
        :param X: (M x 2) M samples of 2 values
        TODO: Support any number of dimensions, settable in the constructor   
        '''
        return np.array([is_stable(self.twip,  [0, 0, x[0], x[1], 0, 0]) for x in X])

def pool_samples(twip, X, n_processes=4, **kwargs):
    ''' pool_samples
    Sample stability of all of M samples contained in X, using multiprocessing
    :param twip: TWIPZi object
    :param X: (M x 2) M samples of 2 values
    :param n_processes (4): Number of processes for multiprocessing pool 
    :return (M) sample label (stable/unstable)
    '''
    Xpart = np.array_split(X, n_processes)
    samp_pool = Pool(n_processes)
    mapping = samp_pool.map(IterSample(twip), Xpart)
    Y = np.hstack(mapping)
    return Y

def classify_points(X, Y, xx, yy, kernel):
    ''' classify_points
    Given a collection of binary labeled samples (X, Y), fit a GaussianProcessClassifier
    and approximate the stable/unstable region for a grid (xx, yy)
    :param X: (M x 2) samples
    :param Y: (M) stable/unstable labels associated with X's samples
    :param xx: (N x N) grid of x values
    :param yy: (N x N) grid of y values
    :param kernel: kernel similarity function
    :return (N x N) probability density estimation
    '''
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, Y)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    Z = gpc.predict_proba(pts)
    Z = Z.reshape((*xx.shape, 2))
    return Z

def sample_attraction_region_mp(twip):
    ''' sample_attraction_region_mp
    Estimate the TWIP region of attraction in two dimensions (uses multiprocessing)
    :param twip: TWIPZi object
    :return ((N x N) stability density estimate, (N x N) grid x values, (N x N) grid y 
            values)
    '''
    # get space filling criteria
    N = 500
    X = np.random.rand(N, 2) * 0.5 - 0.25
    X[:, 1] *= 3
    print("Sampling Initial Pass:")
    Y = pool_samples(twip, X)

    # create a grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # pass through a Gaussian Process
    kernel = RBF(1.0)
    passes = 4
    for i in range(passes):
        N = 200
        Z = classify_points(X, Y, xx, yy, kernel)

        # get new points to sample
        mask = (Z[:, :, 1] < 0.6) & (Z[:, :, 1] > 0.2)
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        new_pts = pts[mask.ravel()]
        np.random.shuffle(new_pts)
        new_pts = new_pts[:N]

        # sample new points
        print("Sampling at Iteration %d/%d:" % (i, passes))
        Ynew = pool_samples(twip, new_pts)

        # update total points
        X = np.vstack((X, new_pts))
        Y = np.hstack((Y, Ynew))

    Z = classify_points(X, Y, xx, yy, kernel)
    return Z, xx, yy


def sample_attraction_region(twip):
    ''' sample_attraction_region
    Estimate the twip's region of attraction in two dimensions
    :param twip: TWIPZi object
    :return ((N x N) stability density estimate, (N x N) grid x values, (N x N) grid y 
            values)
    '''
    # get space filling criteria
    N = 500
    X = np.random.rand(N, 2)*0.5 - 0.25
    X[:, 1] *= 3
    Y = np.zeros(N, dtype=np.bool)
    print("Sampling Initial Pass:")
    for i in tqdm(range(N)):
        x = X[i, :]
        Y[i] = is_stable(twip, [0, 0, x[0], x[1], 0, 0])

    # create a grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # pass through a Gaussian Process
    kernel = RBF(1.0)
    passes = 4
    for i in range(passes):
        N = 200
        Z = classify_points(X, Y, xx, yy, kernel)

        # get new points to sample
        mask = (Z[:, :, 1] < 0.6) & (Z[:, :, 1] > 0.2)
        pts = np.vstack((xx.ravel(), yy.ravel())).T
        new_pts = pts[mask.ravel()]
        np.random.shuffle(new_pts)
        new_pts = new_pts[:N]

        # sample new points
        print("Sampling at Iteration %d/%d:" % (i, passes))
        Ynew = np.zeros(new_pts.shape[0], dtype=np.bool)
        for idx in tqdm(range(new_pts.shape[0])):
            x = new_pts[idx, :]
            Ynew[idx] = is_stable(twip, [0, 0, x[0], x[1], 0, 0])

        # update total points
        X = np.vstack((X, new_pts))
        Y = np.hstack((Y, Ynew))

    Z = classify_points(X, Y, xx, yy, kernel)
    return Z, xx, yy
