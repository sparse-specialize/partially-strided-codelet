import numpy as np


def compute_FOPD(ff, dim):
    di0 = []
    di1 = []
    if dim == 2:
        i0_len = len(ff)
        for i0 in range(i0_len-1):
            i1_len = len(ff[i0])
            i1p1_len = len(ff[i0+1])
            min_both = min(i1_len, i1p1_len)
            tmp = np.zeros(min_both-1, dtype=int)
            for i1 in range(min_both-1):
                tmp[i1] = ff[i0 + 1][i1] - ff[i0][i1]
            di0.append(tmp)
            tmp = np.zeros(i1_len-1, dtype=int)
            for i1 in range(i1_len-1):
                tmp[i1] = ff[i0][i1+1] - ff[i0][i1]
            di1.append(tmp)
    return di0, di1


class codelet:
    def __init__(self, i0, i1, f, g, h):
        # f, g, and h are large functions
        self.i0 = i0
        self.i1 = i1
        self.f = f
        self.g = g
        self.h = h
        self.dfdi0 = []
        self.dfdi1 = []
        self.dgdi0 = []
        self.dgdi1 = []
        self.dhdi0 = []
        self.dhdi1 = []
        self.type = 1
        self.variable_space = 0
        self.num_strided = 0

    def codelet_type(self):
        # Checking all space to fit it into one type.
        dfdi0, dfdi1 = compute_FOPD(self.f, 2)
        dgdi0, dgdi1 = compute_FOPD(self.g, 2)
        dhdi0, dhdi1 = compute_FOPD(self.h, 2)
        # first check the iterations space is equal or not
        for i in self.i0:
            if len(self.f[i]) != len(self.g[i]) != len(self.h[i]):
                # check whether dfdi0 is strided
                self.variable_space = 1  # PSC I
        for i in self.i0:
            if self.dfdi0[i][:] == self.dfdi0[0][0] or self.dfdi1[i][:] == self.dfdi1[0][0]: # or g or h
                # check whether dfdi0 is strided
                self.num_strided += 1  # PSC I



        # then check FOPDs on f

    def get_combination(self):
        s = 1

