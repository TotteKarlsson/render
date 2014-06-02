import sys
import os.path
import json
from bounding_box import BoundingBox
import numpy as np
from features import Features
import itertools
import pylab
from matplotlib import collections  as mc

from L1_utils import load_tilespecs, load_features, load_transforms, save_transforms, extract_features, load_and_transform

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

    for f1, f2 in itertools.combinations(features, 2):
        idx1, idx2, dists = f1.closest_unambiguous_pairs(f2)
        pylab.figure()
        idx1 = idx1[dists <= 40]
        idx2 = idx2[dists <= 40]
        pos1 = f1.locations[idx1, :]
        pos2 = f2.locations[idx2, :]
        lines = [[(y1, x1), (y2, x2)] for (x1, y1), (x2, y2) in zip(pos1, pos2)]
        print features.index(f1), features.index(f2), len(lines)
        lc = mc.LineCollection(lines)
        pylab.gca().add_collection(lc)
        pylab.axis('tight')
        pylab.title('%d %d' % (features.index(f1), features.index(f2)))
    pylab.show()
