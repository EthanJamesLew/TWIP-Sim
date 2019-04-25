'''Math Utilities for System Analysis

Ethan Lew
4/24/19
elew@pdx.edu

'''

import numpy as np

def jacobian(f, fixed_pt):
    # Convert syntax
    x0 = fixed_pt

    # Get number of dimensions of input
    nx = len(fixed_pt)

    # Iteration params
    max_step = 100
    step_ratio = 2.0000001

    # get dimensions of f
    f0 = f(fixed_pt)
    nf = len(f0)

    # Calculate the number of steps
    rel_delta = max_step*np.reciprocal(np.power(step_ratio, np.arange(0, 26, 1)))
    n_steps = len(rel_delta)

    # Initialize the jacobian and the error
    jac = np.zeros((nf, nx))
    err = np.zeros((nf, nx))

    for i in range(0, nx):
        x0_i = x0[i]

        # Calculate the delta
        if (x0_i != 0):
            delta = x0_i*rel_delta
        else:
            delta = rel_delta
    
        # Create a second order approximation
        fdel = np.zeros((nf, n_steps))
        for j in range(0, n_steps):
            fdif = f(swap_element(x0, i, x0_i + delta[j])) - f(swap_element(x0, i, x0_i - delta[j]))
            fdel[:, j] = np.ndarray.flatten(fdif)

        derest = fdel * np.tile(0.5 * np.reciprocal(delta),[nf,1])

        # Use Romberg extrapolation to improve result
        for j in range(0, nf):
            der_romb,errest = rombex_trap(step_ratio,derest[j,:],[2, 4])
            
            nest = len(der_romb)

            trim = np.array([0, 1, 2, nest-3, nest-2, nest-1])
            tags = np.argsort(der_romb)
            der_romb_s = np.sort(der_romb)

            mask = np.ones((nest), dtype=bool)
            mask[trim] = False
            der_romb_s = der_romb_s[mask]
            tags = tags[mask]

            errest = errest[tags]

            err[j,i] = np.min(errest)
            ind = np.argmin(errest)
            jac[j,i] = der_romb_s[ind]
    return jac, err

def swap_element(vec,ind,val):
    vec[ind] = val
    return vec


def vec2mat(vec, n, m):
    x = np.arange(0, m, 1)
    y = np.arange(0, n, 1)

    xv, yv = np.meshgrid(x, y)

    ind = xv + yv
    mat = vec[ind]

    if n == 1:
        mat = np.transpose(mat)

    return mat

def rombex_trap(step_ratio, der_init, rombexpon):
    rombexpon = np.array(rombexpon)
    
    srinv = 1/step_ratio

    nexpon = len(rombexpon)
    rmat = np.ones((nexpon+2, nexpon+1))
 
    rmat[1, 1:3] =  np.power(srinv, rombexpon)
    rmat[2, 1:3] = np.power(srinv, 2*rombexpon)
    rmat[3, 1:3] = np.power(srinv, 3*rombexpon)

    qromb, rromb = np.linalg.qr(rmat)

    ne = len(der_init)
    rhs = vec2mat(der_init, nexpon+2, ne - (nexpon + 2))

    #rombcoefs = np.linalg.lstsq(rromb, np.matmul(np.transpose(qromb), rhs), rcond=None)[0]
    rombcoefs = np.linalg.solve(rromb, np.matmul(np.transpose(qromb), rhs))
    der_romb = np.transpose(rombcoefs[0, :])

    s = np.sqrt(np.sum((rhs - np.matmul(rmat, rombcoefs))**2,axis=0))

    #rinv = np.linalg.lstsq(rromb, np.eye(nexpon + 1), rcond=None)[0]
    rinv = np.linalg.solve(rromb, np.eye(nexpon + 1))
    cov1 = np.sum(rinv**2, axis=1)

    errest = np.transpose(s)*12.7062047361747*np.sqrt(cov1[0])

    return der_romb, errest


if __name__ == "__main__":

    def test_function(c):
        x_data = np.array([[0.0],[0.1],[0.2]])
        y_data = 1+2*np.exp(0.75*x_data)
        return ((c[0] + c[1] * np.exp(c[2]*x_data)) - y_data)**2

    jac, err = jacobian(test_function, [1, 1, 1])
    print(jac)
    print(err)


