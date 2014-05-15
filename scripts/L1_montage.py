import sys
import json
from bounding_box import BoundingBox
import pylab
import numpy as np
import itertools
from scipy.spatial.distance import cdist
from scipy.stats import norm
from weighted_ransac import weighted_ransac
from collections import defaultdict
import prettyplotlib as ppl
import scipy.optimize

import theano.tensor as T
from theano import function
from transform import Transform


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
    locations[:, 0] += delta_x
    locations[:, 1] += delta_y
    return features

def compute_alignments(tilespec_file, feature_file, overlap_frac=0.06, max_diff=0.25):
    bboxes = {ts["mipmapLevels"]["0"]["imageUrl"] : BoundingBox(*ts["bbox"]) for ts in load_tilespecs(tilespec_file)}
    features = {fs["mipmapLevels"]["0"]["imageUrl"] : fs["mipmapLevels"]["0"]["featureList"] for fs in load_features(feature_file)}
    assert set(bboxes.keys()) == set(features.keys())

    # XXX - needed?
    for k in bboxes:
        features[k] = extract_features(features[k])
        features[k] = offset_features(features[k], bboxes[k].from_x, bboxes[k].from_y)

    max_match_distance = 2 * overlap_frac * max(bboxes.values()[-1].shape())

    tilenames = sorted(bboxes.keys())
    fixed_tile = tilenames[0]
    moving_tiles = tilenames[1:]

    # set up transforms
    transforms = {tn : Transform(name=tn, fixed=(tn == fixed_tile)) for tn in tilenames}
    args = [arg for tn in tilenames for arg in transforms[tn].args()]

    dists = []

    tot_matches = 0
    all_good_matches = {}

    for k1, k2 in itertools.combinations(tilenames, 2):
        if not bboxes[k1].overlap(bboxes[k2]): continue

        locs1, features1 = features[k1]
        locs2, features2 = features[k2]

        overlap_bbox = bboxes[k1].intersect(bboxes[k2]).expand(scale=(1 + overlap_frac))
        mask1 = overlap_bbox.contains(locs1)
        mask2 = overlap_bbox.contains(locs2)
        max_good_count = min(mask1.sum(), mask2.sum())

        locs1 = locs1[mask1, :]
        features1 = features1[mask1, :]
        locs2 = locs2[mask2, :]
        features2 = features2[mask2, :]

        image_dists = cdist(locs1, locs2)
        feature_dists = cdist(features1, features2)

        feature_dists[image_dists > max_match_distance] = np.inf

        best_k1_index = np.argmin(feature_dists, axis=0)
        best_k2_index = np.argmin(feature_dists, axis=1)
        cur_good_matches = [(locs1[idx1, :], locs2[idx2, :])
                            for (idx1, idx2) in enumerate(best_k2_index)
                            if (best_k1_index[idx2] == idx1) and
                            (feature_dists[idx1, idx2] < max_diff)]

        if len(cur_good_matches) > 1:
            tot_matches += len(cur_good_matches)
            cur_good_matches = np.array(cur_good_matches)
            all_good_matches[k1, k2] = cur_good_matches
            pt1 = cur_good_matches[:, 0, :]
            pt2 = cur_good_matches[:, 1, :]
            p1 = transforms[k1].transform(pt1)
            p2 = transforms[k2].transform(pt2)

            curdist = T.sqrt(T.sum(T.sqr(p1 - p2), axis=1))
            dists.append(curdist)

    dists = T.concatenate(dists)
    err = T.sum(dists) / tot_matches

    distfun = function(args, dists)
    errfun = function(args, err)
    errgrad = function(args, T.grad(err, args))
    intermediates = []

    def f(x):
        result = errfun(*(x.tolist()))
        intermediates.append((result, x))
        return result
    def g(x):
        result = np.array(errgrad(*(x.tolist())))
        return result

    def callback(x):
        pass

    best_w_b = scipy.optimize.fmin_bfgs(f=f,
                                        x0=np.zeros(len(args)),
                                        fprime=g,
                                        callback=callback,
                                        maxiter=5000,
                                        disp=False)


    rot_trans = best_w_b.tolist()

    print "    Median distance", np.median(distfun(*(best_w_b.tolist())))

    result = []
    for tn in tilenames:
        num_args = len(transforms[tn].args())
        d = transforms[tn].args_to_dict(rot_trans[:num_args])
        d["tile"] = tn
        result.append(d)
        rot_trans = rot_trans[num_args:]
    return result


if __name__ == '__main__':
    res = compute_alignments(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], "wb") as transforms:
        json.dump(res, transforms,
                  sort_keys=True,
                  indent=4)
