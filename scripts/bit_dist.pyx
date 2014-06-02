import numpy as np
cimport numpy as cnp

cdef extern:
        int __builtin_popcountl(size_t) nogil

cdef void _bit_dist(cnp.int64_t[:, :] a,
                    cnp.int64_t[:, :] b,
                    cnp.uint32_t[:, :] dists) nogil:
   cdef int i, j, k, l, count
   l = a.shape[1]

   for i in range(a.shape[0]):
       for j in range(b.shape[0]):
           count = 0
           for k in range(l):
               count += __builtin_popcountl(a[i, k] ^ b[j, k])
           dists[i, j] = count

cdef void _bit_dist(cnp.int64_t[:, :] a,
                    cnp.int64_t[:, :] b,
                    cnp.uint32_t[:] dists) nogil:
   cdef int i, j, k, l, count
   l = a.shape[1]

   for i in range(a.shape[0]):
       count = 0
       for k in range(l):
           count += __builtin_popcountl(a[i, k] ^ b[i, k])
       dists[i] = count

def bit_dist(a, b):
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.uint32)
    _bit_dist(a, b, out)
    return out

def bit_dist_pairwise(a, b):
    assert a.shape == b.shape
    out = np.zeros(a.shape[0], dtype=np.uint32)
    _bit_dist_pairwise(a, b, out)
    return out
