import sys
import os.path
import json
from bounding_box import BoundingBox
import numpy as np
from features import Features
import itertools

from L1_utils import load_tilespecs, load_features, load_transforms, save_transforms, extract_features, load_and_transform

import pyximport; pyximport.install()
from bit_dist import bit_dist

import L1_mosaic_derivs
from hesfree import hessian_free

eps = np.finfo(np.float32).eps

def weight(skip):
    return 1.25 ** -(abs(skip) - 1)


def optimize(pairwise_matches, num_slices, edge_length):
    param_idx = 3 * np.range(num_slices, np.int)

    def dists(param_vec):
        dists = []
        for (k1, k2), matches in all_good_matches.iteritems():
            x1, y1, x2, y2 = matches.T

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]

            R1 /= edge_length
            R2 /= edge_length

            nx1 = np.cos(R1) * x1 - np.sin(R1) * y1 + Tx1
            ny1 = np.sin(R1) * x1 + np.cos(R1) * y1 + Ty1

            nx2 = np.cos(R2) * x2 - np.sin(R2) * y2 + Tx2
            ny2 = np.sin(R2) * x2 + np.cos(R2) * y2 + Ty2

            D = np.sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2 + 1)
            dists.append(D)

        d = np.concatenate(dists)
        return d


    def err_and_gradient(param_vec, noisy=False):
        weighted_dists = []
        g = np.zeros_like(param_vec)
        for (k1, k2), matches in pairwise_matches.iteritems():
            x1, y1, x2, y2 = matches.T

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]

            D, dR1, dTx1, dTy1, dR2, dTx2, dTy2 = \
                L1_mosaic_derivs.f_fprime(x1, y1, R1, Tx1, Ty1,
                                          x2, y2, R2, Tx2, Ty2,
                                          edge_length)
            w = weight(k1 - k2) / matches.shape[0]
            weighted_dists.append(w * D)
            if noisy:
                print k1, g[param_idx[k1]:][:3], (dR1, dTx1, dTy1)
                print k2, g[param_idx[k2]:][:3], (dR2, dTx2, dTy2)
            g[param_idx[k1]:][:3] += (w * dR1, w * dTx1, w * dTy1)
            g[param_idx[k2]:][:3] += (w * dR2, w * dTx2, w * dTy2)

        return sum(weighted_dists), g

    def err(params):
        return err_and_gradient(params)[0]

    def gradient(params):
        return err_and_gradient(params)[1]

    def Hv(param_vec, v):
        Hv = np.zeros_like(param_vec)

        for (k1, k2), matches in pairwise_matches.iteritems():
            x1, y1, x2, y2 = matches.T

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]
            vR1, vTx1, vTy1 = v[param_idx[k1]:][:3]
            vR2, vTx2, vTy2 = v[param_idx[k2]:][:3]

            dvR1, dvTx1, dvTy1, dvR2, dvTx2, dvTy2 = \
                L1_mosaic_derivs.Hv(x1, y1, R1, Tx1, Ty1, vR1, vTx1, vTy1,
                                    x2, y2, R2, Tx2, Ty2, vR2, vTx2, vTy2,
                                    edge_length)
            w = weight(k1 - k2) / matches.shape[0]
            Hv[param_idx[k1]:][:3] += (w * dvR1, w * dvTx1, w * dvTy1)
            Hv[param_idx[k2]:][:3] += (w * dvR2, w * dvTx2, w * dvTy2)

        return Hv


    def f(x):
        return err_and_gradient(x)[0]

    def g(x):
        return err_and_gradient(x)[1]

    def callback(x):
        D = dists(x)
        print np.median(D), D.sum() / tot_matches, np.linalg.norm(g(x))
        pass

    best_params = hessian_free(f=err,
                               x0=np.zeros(3 * num_slices),
                               fprime=gradient,
                               fhessp=Hv,
                               callback=callback,
                               maxiter=100)

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

    # find matches by index
    all_matches = [((idx1, idx2), features[idx1].match(features[idx2]))
                   for idx1, idx2 in itertools.combinations(range(length(features)), 2)]
    print "MATCHES":
    for i1, i2, m in all_matches:
        if m[0].size > 0:
            print "   ", i1, i2, m.size
        else:
            print "   ", i1, i2, "NO MATCHES"

    # extract actual locations
    all_matches = [((idx1, idx2), np.hstack((features[idx1].locations[matches[0], :],
                                             features[idx2].locations[matches[1], :]))) 
                   for (idx1, idx2), matches in all_matches
                   if matches[0].size > 0]

    param_vec = optimize(all_matches, len(features), edge_length)

    fixed_tile = tile_files[0]
    moving_tiles = tile_files[1:]

