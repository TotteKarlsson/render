import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

eps = np.finfo(np.float32).eps

numpts = 100
numlinks = 213
gridsize = (20, 30)

# Nx2
Gx, Gy = np.mgrid[:gridsize[0], :gridsize[1]]
G = np.hstack((Gx.reshape((-1, 1)), Gy.reshape((-1, 1))))

# create a bunch of random weighted combinations
weights = np.random.uniform(size=(numpts, 4))
weights = weights / weights.sum(axis=1).reshape((-1, 1))
grid_indices = np.random.uniform(low=0, high=np.prod(gridsize), size=(numpts, 4)).astype(int)
pt_indices = np.empty((numpts, 4), int)
pt_indices[...] = np.arange(numpts).reshape((-1, 1))
W = sp.csr_matrix((weights.ravel(), (pt_indices.ravel(), grid_indices.ravel())), (numpts, np.prod(gridsize)))

# Create a bunch of linking values

Pa_ = np.random.uniform(low=0, high=numpts, size=(numlinks, 1)).astype(int).ravel()
Pb_ = np.random.uniform(low=0, high=numpts, size=(numlinks, 1)).astype(int).ravel()
Pa = sp.csr_matrix((np.ones_like(Pa_), (np.arange(numlinks, dtype=int), Pa_)), (numlinks, numpts))
Pb = sp.csr_matrix((np.ones_like(Pb_), (np.arange(numlinks, dtype=int), Pb_)), (numlinks, numpts))
P = Pa - Pb

PaW = Pa * W
PbW = Pb * W
M = P * W

# and desired lengths
L = np.sqrt(((M * G) ** 2).sum(axis=1) + eps)
L = L + np.random.uniform(-1, 1, L.shape)
L[L < 0] = 0
L = L.reshape((-1, 1)) * 0

def err(offset=np.zeros_like(G)):
    assert offset.shape == G.shape
    dXY = M * (G + offset)
    lens = np.sqrt((dXY ** 2).sum(axis=1)).reshape((-1, 1))
    assert lens.shape == L.shape
    return np.sum(np.abs(lens - L))

def gradient_err_divided(step):
    def f_i(i, j):
        spike = np.zeros_like(G, np.float64)
        spike[i, j] = step
        e = err(spike)
        return e

    f_0 = err()
    grad = np.zeros_like(G, dtype=np.float)
    grad[:, 0] = [(f_i(i, 0) - f_0) / step for i in range(G.shape[0])]
    grad[:, 1] = [(f_i(i, 1) - f_0) / step for i in range(G.shape[0])]
    return grad

def gradient_err():
    A = PaW * G
    B = PbW * G
    lens = np.sqrt(((A - B) ** 2).sum(axis = 1)).reshape((-1, 1))
    sL = np.sign(lens - L)
    print sL.dtype
    lens[lens == 0] = eps

    # maximum increase for point A is along the B->A vector if current length
    # is greater than desired length
    gXY = M.T * (sL * (A - B) / lens)

    # slightly more verbose:
    # grad_A = sL * AB / lens
    # grad_B = - sL * AB / lens
    # grad_P = Pa.T * grad_A + Pb.T * grad_B
    # gXY = W.T * grad_P

    return gXY

errs = []

dd = gradient_err_divided(0.00001)
sy = gradient_err()
print "ABS", abs(sy - dd).max()
print "REL", (abs(sy - dd) / abs(dd + eps)).max()


