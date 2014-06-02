import sys
import os.path
import json
from bounding_box import BoundingBox
import numpy as np
from features import Features
import itertools
import pylab

from L1_utils import load_tilespecs, load_features, load_transforms, save_transforms, extract_features, load_and_transform

import pyximport; pyximport.install()
from bit_dist import bit_dist

import L1_mosaic_derivs
from hesfree import hessian_free

eps = np.finfo(np.float32).eps

def weight(skip):
    return 1.25 ** -(abs(skip) - 1)


if __name__ == '__main__':
    tile_files = sys.argv[1::4]
    feature_files = sys.argv[2::4]
    input_transforms = sys.argv[3::4]
    output_transforms = sys.argv[4::4]

    tilenames = [os.path.basename(t).split('_')[1] for t in tile_files]

    bboxes, features = zip(*[load_and_transform(tf, ff, it)
                             for tf, ff, it in
                             zip(tile_files, feature_files, input_transforms)])

    # We normalize rotation angles by the edge length, to try to keep
    # translations/rotations on approximately the same scale
    edge_length = max(max(b.shape()) for b in bboxes)

    for f in features:
        f.bin_by_substring()

    for f1, f2 in itertools.combinations(features):
        idx1, idx2, dists = f1.closest_unambiguous_pairs(f2)
        pylab.hist(dists, 100)
        pylab.show()
