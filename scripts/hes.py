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
    lens = np.sqrt((dXY ** 2).sum(axis=1) + 1).reshape((-1, 1))
    assert lens.shape == L.shape
    return np.sum(np.sqrt((lens - L)**2 + 1))

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

def gradient_err(offset=np.zeros_like(G)):
    AB = M * (G + offset)
    lens = np.sqrt(((AB) ** 2).sum(axis = 1) + 1).reshape((-1, 1))
    # maximum increase for point A is along the B->A vector if current length
    # is greater than desired length
    delta_lens = (lens - L)
    dAB = (AB  / lens) * (delta_lens / np.sqrt(delta_lens**2 + 1))
    gXY = M.T * dAB

    # slightly more verbose:
    # grad_A = sL * AB / lens
    # grad_B = - sL * AB / lens
    # grad_P = Pa.T * grad_A + Pb.T * grad_B
    # gXY = W.T * grad_P

    return gXY

def Hv_divided(v, step):
    g1 = gradient_err(v * step)
    g2 = gradient_err(-v * step)
    return (g1 - g2) / (2 * step)

def Hv(v):
    # compute as d(gradient(G + e * v)) / de at e == 0
    AB = M * G
    Mv = M * v
    # dAB / de = Mv

    step = 1e-7
    ABe = M * (G + step * v)

    lens = np.sqrt((AB ** 2).sum(axis = 1) + 1).reshape((-1, 1))
    lense = np.sqrt((ABe ** 2).sum(axis = 1) + 1).reshape((-1, 1))


    normed_AB = AB / lens
    normed_ABe = ABe / lense

    # d [sqrt(x**2 + y**2  + 1)] / dx = x / sqrt(x**2 + y**2  + 1)
    d_lens_de = (Mv * normed_AB).sum(axis=1).reshape((-1, 1))
    # quotient rule
    d_NAB_de = ((lens * Mv) - (AB * d_lens_de)) / lens**2

    delta_lens = (lens - L)
    # d delta_lens / de = d lens / de

    normed_delta_lens = delta_lens / np.sqrt(delta_lens**2 + 1)
    # d normed_delta_lens / de = (1 / (np.sqrt(delta_lens**2 + 1))**3) * (d delta_lens / de)
    d_NDL_de = d_lens_de / np.sqrt(delta_lens**2 + 1)**3

    d_AB_de = normed_AB * (d_NDL_de) + normed_delta_lens * d_NAB_de
    return M.T * d_AB_de

errs = []

step = 1e-5
dd = gradient_err_divided(step)
sy = gradient_err()
print "ABS diff gradient", abs(sy - dd).max()
print "REL diff gradient", (abs(sy - dd) / abs(dd + eps)).max()


v = np.random.uniform(-1, 1, G.shape)
dd = Hv_divided(v, step)
sy = Hv(v)
print "ABS diff Hv", abs(sy - dd).max()
print "REL diff Hv", (abs(sy - dd) / abs(dd + eps)).max()
