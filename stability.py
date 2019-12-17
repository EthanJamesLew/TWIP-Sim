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

def is_stable(twip, IC, t_max=10, inner_radius=0.18, outer_radius=1):
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


def classify_points(X, Y, xx, yy, kernel):
    gpc = GaussianProcessClassifier(kernel=kernel).fit(X, Y)
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    Z = gpc.predict_proba(pts)
    Z = Z.reshape((*xx.shape, 2))
    return Z

def sample_attraction_region(twip):
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
