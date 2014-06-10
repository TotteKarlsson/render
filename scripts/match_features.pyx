import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from cython import boundscheck, wraparound

cdef extern from "math.h":
    float sqrt(float x) nogil

@boundscheck(False)
@wraparound(False)
cdef float norm2(float *a, float *b, int l, float top) nogil:
   cdef float diff
   cdef int i
   diff = 0.0
   for i in range(l):
       diff = diff + (a[i] - b[i]) ** 2
       if diff > top:
           return diff
   return diff

@boundscheck(False)
@wraparound(False)
cdef void find_best_matches_oneway(cnp.float32_t[:, :] locs_1, cnp.float32_t[:, :] features_1,
                                   cnp.float32_t[:, :] locs_2, cnp.float32_t[:, :] features_2,
                                   cnp.uint32_t[:] match_indices,
                                   cnp.float32_t[:] match_diffs,
                                   float max_dist) nogil:
   cdef int i, j, k, best_match, l
   cdef float diff, best_diff

   max_dist = max_dist * max_dist

   l = features_1.shape[1]
   for i in prange(locs_1.shape[0], nogil=True):
       best_diff = 1.0
       best_match = 0
       for j in range(locs_2.shape[0]):
           if ((locs_1[i, 0] - locs_2[j, 0]) ** 2 +
               (locs_1[i, 1] - locs_2[j, 1]) ** 2) >= max_dist:
               continue
           diff = norm2(& features_1[i, 0], & features_2[j, 0], l, best_diff)
           if diff < best_diff:
               best_diff = diff
               best_match = j
       match_diffs[i] = sqrt(best_diff)
       match_indices[i] = best_match

def match_features(locs_1, features_1,
                   locs_2, features_2,
                   max_distance):
    assert locs_1.shape[0] == features_1.shape[0]
    assert locs_2.shape[0] == features_2.shape[0]
    assert locs_1.shape[1] == 2
    assert locs_2.shape[1] == 2
    assert features_1.shape[1] == features_2.shape[1]

    match_indices_1 = np.zeros((locs_1.shape[0]), dtype=np.uint32)
    match_indices_2 = np.zeros((locs_2.shape[0]), dtype=np.uint32)
    # set to maximum difference to start
    match_diffs_1 = np.ones((locs_1.shape[0]), dtype=np.float32)
    match_diffs_2 = np.ones((locs_2.shape[0]), dtype=np.float32)

    find_best_matches_oneway(locs_1, features_1,
                             locs_2, features_2,
                             match_indices_1, match_diffs_1,
                             max_distance)

    find_best_matches_oneway(locs_2, features_2,
                             locs_1, features_1,
                             match_indices_2, match_diffs_2,
                             max_distance)

    return match_indices_1, match_indices_2, match_diffs_1
