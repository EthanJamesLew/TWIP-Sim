'''
Ethan Lew
(elew@pdx.edu)

Run TWIP stability anaylsis. The stability is determined by several assumptions:
    1. The stable region is connected
    2. The stable region is stable at 0

The algorithm operates by:
    1. Precompute the largest uniform hypercube centered at 0 to be stable. This step accelerates the simulations by
    allowing them to terminate earlier.
    2. Construct a simulation domain in the state space, using a hyper-rectangular search tree. Do this by:
        Sufficiently sampling near the boundaries to determine if the corners are stable
            1. Label the hyper-rectangle as STABLE
            2. Subdivide the hyper-rectangle and determine if any of them are stable
'''

import numpy as np
from robot import PIDRobot
from hyper_tree import HyperRectTree

class StableTwip:
    def __init__(self, twip):
        self._twip = twip
        self._stable_box = None
        self._samples = 0
        l = 1
        self.tree = HyperRectTree([-l, -l, -l, l, l, l])

    def is_stable(self, IC, t_max = 30, stable_box = [0, 0, 0]):
        self._samples += 1
        dt = 1/30
        self._twip.set_IC(IC)
        self._twip.update_current_state(dt, [0,0,0,0])
        t = 0
        while(t < t_max):
            self._twip.update_current_state(dt, [0, 0, 0, 0])
            t += dt
            cpos = self._twip.twip.q
            if((abs(cpos[2]) < stable_box[0]) and (abs(cpos[5]) < stable_box[1]) and (abs(cpos[3]) < stable_box[2])):
                return True
            tree_sample = self.tree.get_cell([cpos[2], cpos[5], cpos[3]])
            if tree_sample is not None:
                if tree_sample.get_label() is True:
                    return True
                elif tree_sample.get_label() is False:
                    return False
            if(abs(cpos[5]) > 2):
                return False
        return True

    def find_stable_box(self, err=0.005):
        print("Finding stable region....")
        # Find a stable box
        norm_err = 1000
        err_tol = err
        box = np.array([10, 10, 10])
        last_box = box * 0
        stab = False
        while ((norm_err > err_tol) or not stab):
            stab = self.is_stable([0, 0, box[0], box[2], 0, box[1]])
            if (stab):
                box = box + abs(last_box - box) / 2
                norm_err = np.linalg.norm(abs(last_box - box))
                last_box = box
            else:
                box = box - abs(last_box - box) / 2
        print("Found ", box)
        self._stable_box = box
        return box

    def determine_cell(self, cell, max_sample_fact=6):
        # get samples
        max_sample = int(np.ceil(np.linalg.norm(np.array(cell.coords[:3]) - np.array(cell.coords[3:]))*max_sample_fact))
        samps = [cell.get_rand_in_cell() for i in range(0, max_sample)]
        # determines whether ALL of cell is stable
        is_stable = True

        # track samples
        curr = None
        prev = None

        # start sampling
        for idx, samp in enumerate(samps):
            # determine whether current sample is stable
            if not self.is_stable([0, 0, samp[0], samp[2], 0, samp[1]]):
                curr = False
                is_stable = False
            else:
                curr = True

            if (curr is not None) and (prev is not None):
                # cell has both stable and unstable regions -- SUBDIVIDE
                if curr is not prev:
                    return None

            prev = curr
        print("Found consistent cell: ", cell, 'STABLE' if cell.get_label() else 'UNSTABLE')
        # cell is consistent - do NOT subdivide
        return is_stable

    def find_stable_region(self, bounds=[10, 10, 10]):
        if self._stable_box is None:
            self.find_stable_box()
        self.tree.subdivide()
        for i in range(2):
            cells_m = self.tree.get_all()
            cells_m = np.reshape(cells_m, (len(cells_m)//self.tree.n, self.tree.n))
            search_order = np.argsort(np.linalg.norm(cells_m, axis=1))
            cells_m = cells_m[search_order, :].flatten()
            for jj in range(0, len(cells_m)//self.tree.n):
                cell_m = cells_m[jj*self.tree.n:(jj+1)*self.tree.n]
                cell = self.tree.get_cell(cell_m)
                print(cell)
                label = cell.get_label()
                if label is None:
                    state = self.determine_cell(cell)
                    if state is None:
                        cell.subdivide()
                    else:
                        cell.set_label(state)

if __name__ == "__main__":
    twip = PIDRobot(0.03)
    st = StableTwip(twip)
    st._stable_box = [1.875, 1.875, 1.875]

    st.find_stable_region()








