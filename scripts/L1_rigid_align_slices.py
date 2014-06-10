import sys
import os.path
import json
from bounding_box import BoundingBox
import numpy as np
from features import Features
import itertools
import time



class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from L1_utils import load_tilespecs, load_features, load_transforms, save_transforms, extract_features, load_and_transform

import pyximport; pyximport.install()
from bit_dist import bit_dist

import L1_mosaic_derivs
from hesfree import hessian_free

eps = np.finfo(np.float32).eps

def weight(skip, num_matches):
    return (1.25 ** -(abs(skip) - 1)) / num_matches


def optimize(pairwise_matches, num_slices, edge_length):

    # parameter vector is R, Tx, Ty for each slice.
    # create an index into the parameter vector for each match.
    indices_R1 = np.concatenate([3 * k1 * np.ones(matches.shape[0], dtype=np.int) for (k1, k2), matches in pairwise_matches])
    indices_Tx1 = 1 + indices_R1
    indices_Ty1 = 2 + indices_R1
    indices_R2 = np.concatenate([3 * k2 * np.ones(matches.shape[0], dtype=np.int) for (k1, k2), matches in pairwise_matches])
    indices_Tx2 = 1 + indices_R2
    indices_Ty2 = 2 + indices_R2

    x1 = np.concatenate([matches[:, 0] for _, matches in pairwise_matches])
    y1 = np.concatenate([matches[:, 1] for _, matches in pairwise_matches])
    x2 = np.concatenate([matches[:, 2] for _, matches in pairwise_matches])
    y2 = np.concatenate([matches[:, 3] for _, matches in pairwise_matches])
    weights = np.concatenate([weight(k1 - k2, matches.shape[0]) * np.ones(matches.shape[0])
                              for (k1, k2), matches in pairwise_matches])

    param_len = 3 * num_slices
    num_pairs = len(pairwise_matches)

    def dists(param_vec):
        R1, Tx1, Ty1 = [param_vec[idx] for idx in [indices_R1, indices_Tx1, indices_Ty1]]
        R2, Tx2, Ty2 = [param_vec[idx] for idx in [indices_R2, indices_Tx2, indices_Ty2]]
        R1 /= edge_length
        R2 /= edge_length

        c1 = np.cos(R1)
        s1 = np.sin(R1)
        c2 = np.cos(R2)
        s2 = np.sin(R2)

        nx1 = c1 * x1 - s1 * y1 + Tx1
        ny1 = s1 * x1 + c1 * y1 + Ty1

        nx2 = c2 * x2 - s2 * y2 + Tx2
        ny2 = s2 * x2 + c2 * y2 + Ty2

        D = np.sqrt((nx1 - nx2)**2 + (ny1 - ny2)**2 + 1)
        return D

    prev = [-1, 0]
    def err_and_gradient(param_vec, noisy=False):
        # cache previous evaluation
        if np.all(param_vec == prev[0]):
            return prev[1]

        # pull
        R1, Tx1, Ty1 = [param_vec[idx] for idx in [indices_R1, indices_Tx1, indices_Ty1]]
        R2, Tx2, Ty2 = [param_vec[idx] for idx in [indices_R2, indices_Tx2, indices_Ty2]]
        D, dR1, dTx1, dTy1, dR2, dTx2, dTy2 = \
            L1_mosaic_derivs.f_fprime(x1, y1, R1, Tx1, Ty1,
                                      x2, y2, R2, Tx2, Ty2,
                                      edge_length)

        wD = sum(D * weights) / num_pairs
        # push
        g = sum([np.bincount(idx, vals * weights, minlength=param_len)
                 for idx, vals in zip([indices_R1, indices_Tx1, indices_Ty1,
                                       indices_R2, indices_Tx2, indices_Ty2],
                                      [dR1, dTx1, dTy1, dR2, dTx2, dTy2])])
        g /= num_pairs

        # udpate cache
        prev[0] = param_vec.copy()
        prev[1] = (wD, g)

        return wD, g

    def err(params):
        return err_and_gradient(params)[0]

    def gradient(params):
        return err_and_gradient(params)[1]

    def Hv(param_vec, v):
        # pull
        R1, Tx1, Ty1 = [param_vec[idx] for idx in [indices_R1, indices_Tx1, indices_Ty1]]
        R2, Tx2, Ty2 = [param_vec[idx] for idx in [indices_R2, indices_Tx2, indices_Ty2]]
        vR1, vTx1, vTy1 = [v[idx] for idx in [indices_R1, indices_Tx1, indices_Ty1]]
        vR2, vTx2, vTy2 = [v[idx] for idx in [indices_R2, indices_Tx2, indices_Ty2]]
        
        dvR1, dvTx1, dvTy1, dvR2, dvTx2, dvTy2 = L1_mosaic_derivs.Hv(x1, y1, R1, Tx1, Ty1, vR1, vTx1, vTy1,
                                                                     x2, y2, R2, Tx2, Ty2, vR2, vTx2, vTy2,
                                                                     edge_length)
        # push
        Hv = sum([np.bincount(idx, vals * weights, minlength=param_len)
                  for idx, vals in zip([indices_R1, indices_Tx1, indices_Ty1,
                                        indices_R2, indices_Tx2, indices_Ty2],
                                       [dvR1, dvTx1, dvTy1, dvR2, dvTx2, dvTy2])])
        return Hv / num_pairs


    def f(x):
        return err_and_gradient(x)[0]

    def g(x):
        return err_and_gradient(x)[1]

    def callback(x):
        D = dists(x)
        print "25/50/75", np.percentile(D, 25), np.median(D), np.percentile(D, 75), "mean", D.mean(), "F", f(x), np.linalg.norm(g(x))
        pass

    best_params = np.zeros(3 * num_slices)
    print "START",
    # temporarily reset limits
    callback(best_params)

    best_params = hessian_free(f=err,
                               x0=best_params,
                               fprime=gradient,
                               fhessp=Hv,
                               callback=callback,
                               maxiter=100)

    print "END",
    callback(best_params)

    return best_params

if __name__ == '__main__':
    output_transforms_file = sys.argv.pop()
    tile_files = sys.argv[1::3]
    feature_files = sys.argv[2::3]
    input_transforms = sys.argv[3::3]

    assert len(tile_files) == len(feature_files)
    assert len(tile_files) == len(input_transforms)

    tilenames = [os.path.basename(t).split('_')[1] for t in tile_files]

    # don't load all the tiles at once
    # This code relies on itertools.combinations() producing results in sorted order
    max_gap = 6
    cache = {}
    def load_features(idx):
        # flush anything more than 2 * max_gap back
        if (idx - 2 * max_gap) in cache:
            del cache[idx - 2 * max_gap]
        if idx not in cache:
            bbox, features = load_and_transform(tile_files[idx],
                                                feature_files[idx],
                                                input_transforms[idx])
            cache[idx] = tilenames[idx], bbox, features
        return cache[idx]

    max_edge_length = 0
    all_matches = []

    try:
        with open("cached_matches.json", "r") as fp:
            all_matches = json.load(fp)
            all_matches = [(k, np.array(a)) for k, a in all_matches if len(a) > 0]
        max_edge_length = 3 * 16000 - 2 * 0.06 * 16000
    except Exception:
        for sl1, sl2 in itertools.combinations(range(len(tile_files)), 2):
            if abs(sl1 - sl2) > max_gap: continue

            tn1, bbox1, features1 = load_features(sl1)
            tn2, bbox2, features2 = load_features(sl2)

            print "matching slice %d to %d (%d x %d)" % (sl1, sl2, features1.size, features2.size)
            idx1, idx2, diffs, dists = features1.match(features2)
            # ignore overly matchy slices
            trim_size = min(25, (idx1.size / 2))

            # only keep if we have at least 10 matches
            if trim_size > 5:
                # trim to most dense matches in spatial distance
                sorted_dists = np.sort(dists)
                widths = sorted_dists[trim_size:] - sorted_dists[:-trim_size]
                best_idx = widths.argmin()
                lo = sorted_dists[best_idx]
                hi = sorted_dists[best_idx + trim_size]
                mask = (dists > lo) & (dists <= hi)
                idx1 = idx1[mask]
                idx2 = idx2[mask]
                dists = dists[mask]
                print "    trimmed to ", mask.sum(), "MED", np.median(dists), "MAD", np.median(abs(dists - np.median(dists)))

                # extract actual locations
                if mask.sum() > 0:
                    all_matches.append(((sl1, sl2),
                                        np.hstack((features1.locations[idx1, :],
                                                   features2.locations[idx2, :]))))


            # We normalize rotation angles by the edge length, to try to keep
            # translations/rotations on approximately the same scale
            max_edge_length = max([max_edge_length] + list(bbox1.shape()) + list(bbox2.shape()))


        with open("cached_matches.json", "w") as fp:
            json.dump(all_matches, fp,
                      cls=NumpyAwareJSONEncoder)

    print len(all_matches), "pairs from", len(tile_files), "slices"
    print "max edge", max_edge_length
    print "MATCHES:"
    for (i1, i2), m in all_matches:
        if m.shape[0] > 0:
            print "   ", i1, i2, m.shape[0]
        else:
            print "   ", i1, i2, "NO MATCHES"

    param_vec = optimize(all_matches, len(tile_files), max_edge_length)

    def transform_vals(idx):
        if idx == 0:
            return {"rotation_rad" : 0, "trans" : (0, 0)}
        R0, Tx1, Ty1 = param_vec[0:3]
        Ri, Txi, Tyi = param_vec[3*idx:3*idx+3]
        R0 /= max_edge_length
        Ri /= max_edge_length
        c = np.cos(R0)
        s = np.sin(R0)
        Ri -= R0
        Txi_new = c * Txi - s * Tyi - Tx1
        Tyi_new = s * Txi + c * Tyi - Ty1
        return {"rotation_rad" : Ri,
                "trans" : (Txi_new, Tyi_new)}

    output = {tn : transform_vals(idx) for idx, tn in enumerate(tilenames)}

    with open(output_transforms_file, "w") as outf:
        json.dump(output, outf,
                  sort_keys=True,
                  indent=4)
