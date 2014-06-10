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

def weight(skip):
    return 1.25 ** -(abs(skip) - 1)


def optimize(pairwise_matches, num_slices, edge_length,
             block_size=75, block_step=50):

    param_idx = 3 * np.arange(num_slices, dtype=np.int)

    # we optimize a few slices at a time, keeping the first slice fixed, then shifting forward

    # initially, only the first slice is fixed
    solve_low = 1
    solve_high = block_size

    def solve_windowed(all_matches):
        for (k1, k2), matches in all_matches:
            # ignore pairs where both sides are fixed
            if (k1 < solve_low) and (k2 < solve_low):
                continue
            # ignore anything even partially outside our solve window
            if (k1 > solve_high) or (k2 > solve_high):
                continue

            yield (k1, k2), matches

    def dists(param_vec):
        # compute the distances within the current block

        dists = []
        for (k1, k2), matches in solve_windowed(pairwise_matches):
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

    def plot(param_vec):
        import pylab as plt
        from matplotlib.collections import LineCollection

        for (k1, k2), matches in pairwise_matches:
            x1, y1, x2, y2 = matches.T

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]

            R1 /= edge_length
            R2 /= edge_length

            nx1 = np.cos(R1) * x1 - np.sin(R1) * y1 + Tx1
            ny1 = np.sin(R1) * x1 + np.cos(R1) * y1 + Ty1

            nx2 = np.cos(R2) * x2 - np.sin(R2) * y2 + Tx2
            ny2 = np.sin(R2) * x2 + np.cos(R2) * y2 + Ty2

            plt.figure()
            segments = np.concatenate([np.dstack((ny1.reshape((-1, 1)), nx1.reshape((-1, 1)))),
                                       np.dstack((ny2.reshape((-1, 1)), nx2.reshape((-1, 1))))],
                                      axis=1)
            lc = LineCollection(segments)
            plt.gca().add_collection(lc)
            plt.axis('tight')
            plt.title('%d %d' % (k1, k2))
        plt.show()


    prev = [(-1, (-1, -1)), 0]
    def err_and_gradient(param_vec, noisy=False):
        # cache previous evaluation
        prev_param, prev_low_high = prev[0]
        if np.all(param_vec == prev_param) and \
                (solve_low, solve_high) == prev_low_high:
            return prev[1]

        weighted_dists = []
        g = np.zeros_like(param_vec)
        num_pairs = 0
        for (k1, k2), matches in solve_windowed(pairwise_matches):
            num_pairs += 1
            x1, y1, x2, y2 = matches.T

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]

            D, dR1, dTx1, dTy1, dR2, dTx2, dTy2 = \
                L1_mosaic_derivs.f_fprime(x1, y1, R1, Tx1, Ty1,
                                          x2, y2, R2, Tx2, Ty2,
                                          edge_length)
            w = weight(k1 - k2) / x1.size
            weighted_dists.append(w * D)
            if noisy:
                print k1, g[param_idx[k1]:][:3], (dR1, dTx1, dTy1)
                print k2, g[param_idx[k2]:][:3], (dR2, dTx2, dTy2)

            if k1 >= solve_low:
                g[param_idx[k1]:][:3] += (w * dR1, w * dTx1, w * dTy1)
            if k2 >= solve_low:
                g[param_idx[k2]:][:3] += (w * dR2, w * dTx2, w * dTy2)

        wD = sum(weighted_dists) / num_pairs
        g /= num_pairs

        prev[0] = param_vec.copy(), (solve_low, solve_high)
        prev[1] = wD, g
        return wD, g

    def err(params):
        return err_and_gradient(params)[0]

    def gradient(params):
        return err_and_gradient(params)[1]

    def Hv(param_vec, v):
        Hv = np.zeros_like(param_vec)

        num_pairs = 0
        for (k1, k2), matches in solve_windowed(pairwise_matches):
            if (k1 < solve_low) or (k2 < solve_low):
                continue
            x1, y1, x2, y2 = matches.T
            num_pairs += 1

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]
            vR1, vTx1, vTy1 = v[param_idx[k1]:][:3]
            vR2, vTx2, vTy2 = v[param_idx[k2]:][:3]

            dvR1, dvTx1, dvTy1, dvR2, dvTx2, dvTy2 = \
                L1_mosaic_derivs.Hv(x1, y1, R1, Tx1, Ty1, vR1, vTx1, vTy1,
                                    x2, y2, R2, Tx2, Ty2, vR2, vTx2, vTy2,
                                    edge_length)
            w = weight(k1 - k2) / x1.size
            Hv[param_idx[k1]:][:3] += (w * dvR1, w * dvTx1, w * dvTy1)
            Hv[param_idx[k2]:][:3] += (w * dvR2, w * dvTx2, w * dvTy2)

        return Hv / num_pairs


    def f(x):
        return err_and_gradient(x)[0]

    def g(x):
        return err_and_gradient(x)[1]

    def callback(x):
        D = dists(x)
        print "block", solve_low, "to", solve_high, ":", np.percentile(D, 25), np.median(D), D.mean(), np.linalg.norm(g(x))
        pass

    best_params = np.zeros(3 * num_slices)
    print "START",
    # temporarily reset limits
    solve_low = 1
    solve_high = num_slices - 1

    callback(best_params)

    solve_low = 1
    solve_high = block_size

    while True:
        best_params = hessian_free(f=err,
                                   x0=best_params,
                                   fprime=gradient,
                                   fhessp=Hv,
                                   callback=callback,
                                   maxiter=50)
        if (solve_high >= num_slices):
            break
        # update solve window
        solve_low += block_step
        solve_high += block_step

    print "END",
    solve_low = 1
    solve_high = num_slices - 1
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
