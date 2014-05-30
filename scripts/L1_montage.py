import sys
import json
from bounding_box import BoundingBox
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import scipy.optimize

import pyximport; pyximport.install()

import L1_mosaic_derivs
from hesfree import hessian_free

from bit_dist import bit_dist


def rc(filename):
    return filename.split('/')[-1][5:][:5]

def load_tilespecs(file):
    with open(file) as fp:
        return json.load(fp)

def load_features(file):
    with open(file) as fp:
        return json.load(fp)

def extract_features(features):
    locations = np.vstack([np.array([f["location"] for f in features])])
    npfeatures = np.vstack([np.array([f["descriptor"] for f in features])])
    return locations, npfeatures

def offset_features(features, delta_x, delta_y):
    locations, _ = features
    if locations.size > 0:
        locations[:, 0] += delta_x
        locations[:, 1] += delta_y
    return features

def compute_alignments(tilespec_file, feature_file, overlap_frac=0.06, max_diff=32):
    bboxes = {ts["mipmapLevels"]["0"]["imageUrl"] : BoundingBox(*ts["bbox"]) for ts in load_tilespecs(tilespec_file)}
    features = {fs["mipmapLevels"]["0"]["imageUrl"] : fs["mipmapLevels"]["0"]["featureList"] for fs in load_features(feature_file)}
    assert set(bboxes.keys()) == set(features.keys())

    for k in bboxes:
        features[k] = extract_features(features[k])
        features[k] = offset_features(features[k], bboxes[k].from_x, bboxes[k].from_y)

    max_match_distance = 2 * overlap_frac * max(bboxes.values()[-1].shape())

    tilenames = sorted(bboxes.keys())
    fixed_tile = tilenames[0]
    moving_tiles = tilenames[1:]


    # We normalize rotation angles by the edge length, to try to keep
    # translations/rotations on approximately the same scale
    edge_length = max(bboxes[fixed_tile].shape())


    param_idx = {t : 3 * i for i, t in enumerate(tilenames)}

    tot_matches = 0
    all_good_matches = {}

    print "Matches:"
    for k1, k2 in itertools.combinations(tilenames, 2):
        if not bboxes[k1].overlap(bboxes[k2]): continue

        locs1, features1 = features[k1]
        locs2, features2 = features[k2]
        if (locs1.size == 0) or (locs2.size == 0):
            continue

        overlap_bbox = bboxes[k1].intersect(bboxes[k2]).expand(scale=(1 + overlap_frac))
        mask1 = overlap_bbox.contains(locs1)
        mask2 = overlap_bbox.contains(locs2)
        max_good_count = min(mask1.sum(), mask2.sum())

        locs1 = locs1[mask1, :]
        features1 = features1[mask1, :]
        locs2 = locs2[mask2, :]
        features2 = features2[mask2, :]

        image_dists = cdist(locs1, locs2)
        # feature_dists = cdist(features1, features2)
        feature_dists = bit_dist(features1, features2)


        feature_dists[image_dists > max_match_distance] = max_diff + 1

        best_k1_index = np.argmin(feature_dists, axis=0)
        best_k2_index = np.argmin(feature_dists, axis=1)
        cur_good_matches = [(locs1[idx1, 0], locs1[idx1, 1], locs2[idx2, 0], locs2[idx2, 1])
                            for (idx1, idx2) in enumerate(best_k2_index)
                            if (best_k1_index[idx2] == idx1) and
                            (feature_dists[idx1, idx2] < max_diff)]

        if len(cur_good_matches) > 0:
            print rc(k1), rc(k2), len(cur_good_matches)
            tot_matches += len(cur_good_matches)
            all_good_matches[k1, k2] = np.array(cur_good_matches)

    print "   total:", tot_matches

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
        d.sort()
        return d

    def err_and_gradient(param_vec, noisy=False):
        dists = []
        g = np.zeros_like(param_vec)
        for (k1, k2), matches in all_good_matches.iteritems():
            x1, y1, x2, y2 = matches.T

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]

            D, dR1, dTx1, dTy1, dR2, dTx2, dTy2 = \
                L1_mosaic_derivs.f_fprime(x1, y1, R1, Tx1, Ty1,
                                          x2, y2, R2, Tx2, Ty2,
                                          edge_length)
            dists.append(D)
            if noisy:
                print k1, g[param_idx[k1]:][:3], (dR1, dTx1, dTy1)
                print k2, g[param_idx[k2]:][:3], (dR2, dTx2, dTy2)
            g[param_idx[k1]:][:3] += (dR1, dTx1, dTy1)
            g[param_idx[k2]:][:3] += (dR2, dTx2, dTy2)

        return sum(dists) / tot_matches, g / tot_matches

    def err(params):
        return err_and_gradient(params)[0]

    def gradient(params):
        return err_and_gradient(params)[1]

    def Hv(param_vec, v):
        Hv = np.zeros_like(param_vec)

        for (k1, k2), matches in all_good_matches.iteritems():
            x1, y1, x2, y2 = matches.T

            R1, Tx1, Ty1 = param_vec[param_idx[k1]:][:3]
            R2, Tx2, Ty2 = param_vec[param_idx[k2]:][:3]
            vR1, vTx1, vTy1 = v[param_idx[k1]:][:3]
            vR2, vTx2, vTy2 = v[param_idx[k2]:][:3]

            dvR1, dvTx1, dvTy1, dvR2, dvTx2, dvTy2 = \
                L1_mosaic_derivs.Hv(x1, y1, R1, Tx1, Ty1, vR1, vTx1, vTy1,
                                    x2, y2, R2, Tx2, Ty2, vR2, vTx2, vTy2,
                                    edge_length)
            Hv[param_idx[k1]:][:3] += (dvR1, dvTx1, dvTy1)
            Hv[param_idx[k2]:][:3] += (dvR2, dvTx2, dvTy2)

        return Hv / tot_matches


    def f(x):
        return err_and_gradient(x)[0]

    def g(x):
        return err_and_gradient(x)[1]

    def callback(x):
        D = dists(x)
        print np.median(D), D.sum() / tot_matches, np.linalg.norm(g(x))
        pass

    best_params = hessian_free(f=err,
                               x0=np.zeros(3 * len(tilenames)),
                               fprime=gradient,
                               fhessp=Hv,
                               callback=callback,
                               maxiter=100)
    result = []

    for tn in tilenames:
        R, Tx, Ty = best_params[param_idx[tn]:][:3]
        d = {"tile" : tn,
             "rotation_rad" : R / edge_length,
             "trans" : [Tx, Ty]}
        result.append(d)

    base = [r for r in result if r["tile"] == fixed_tile][0]
    base_R = base["rotation_rad"]
    base_Tx, base_Ty = base["trans"]
    c = np.cos(- base_R)
    s = np.sin(- base_R)
    transforms = {}
    for r in result:
        r["rotation_rad"] -= base_R
        tmp_x, tmp_y = r["trans"]
        new_x = c * tmp_x - s * tmp_y - base_Tx
        new_y = s * tmp_x + c * tmp_y - base_Ty
        r["trans"] = new_x, new_y
        transforms[r["tile"]] = r["rotation_rad"], new_x, new_y

    if False:
        import pylab
        for tn in tilenames:
            R, Tx, Ty = transforms[tn]
            c = np.cos(R)
            s = np.sin(R)
            xpos = features[tn][0][:, 0]
            ypos = features[tn][0][:, 1]
            newx = c * xpos - s * ypos + Tx
            newy = s * xpos + c * ypos + Ty
            pylab.plot(newy, newx, '.')
        pylab.show()

    rots = np.array([d["rotation_rad"] for d in result])
    print "max abs rotation", np.max(np.abs(rots - np.mean(rots)))

    return result


if __name__ == '__main__':
    print "Mosaicing", sys.argv[1]
    res = compute_alignments(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], "wb") as transforms:
        json.dump(res, transforms,
                  sort_keys=True,
                  indent=4)
    print
