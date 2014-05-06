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


alpha = 0.1

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

def transform(R, Tx, Ty, p):
    if R is 0:  # special case for fixed tile
        return p[0], p[1]
    s = T.sin(R / 360)
    c = T.cos(R / 360)
    new_x =   c * p[0] + s * p[1] + Tx
    new_y = - s * p[0] + c * p[1] + Ty
    return new_x, new_y

def transform_np(R, Tx, Ty, p):
    s = np.sin(R / 360)
    c = np.cos(R / 360)
    new_x =   c * p[0] + s * p[1] + Tx
    new_y = - s * p[0] + c * p[1] + Ty
    return new_x, new_y

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

    # set up theano variables for rotations and translations
    rots = {k : T.dscalar('rot %s' % rc(k)) for k in features}
    trans_x = {k : T.dscalar('trans_x %s' % rc(k)) for k in features}
    trans_y = {k : T.dscalar('trans_y %s' % rc(k)) for k in features}

    # fix first tile
    fixed_tile = tilenames[0]
    moving_tiles = tilenames[1:]
    rots[fixed_tile] = 0
    trans_x[fixed_tile] = 0
    trans_y[fixed_tile] = 0
    print "fixed", fixed_tile

    err = 0

    tot_matches = 0
    all_good_matches = {}

    for k1, k2 in itertools.combinations(bboxes.keys(), 2):
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
            print len(cur_good_matches)
            tot_matches += len(cur_good_matches)
            cur_good_matches = np.array(cur_good_matches)
            all_good_matches[k1, k2] = cur_good_matches
            new_x_1, new_y_1 = transform(rots[k1], trans_x[k1], trans_y[k1], (cur_good_matches[:, 0, 0], cur_good_matches[:, 0, 1]))
            new_x_2, new_y_2 = transform(rots[k2], trans_x[k2], trans_y[k2], (cur_good_matches[:, 1, 0], cur_good_matches[:, 1, 1]))
            err = err + T.sum(T.sqrt(T.sqr(new_x_1 - new_x_2) + T.sqr(new_y_1 - new_y_2)))

    args = [rots[k] for k in moving_tiles] + \
        [trans_x[k] for k in moving_tiles] + \
        [trans_y[k] for k in moving_tiles]

    err = err / tot_matches

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
                                      full_output=True,
                                      disp=1)


    for idx, (err, rot_trans) in enumerate(intermediates):
        print "writing", idx, len(intermediates)
        pylab.figure(figsize=(8.0, 5.0))

        rots = {k : i for k, i in zip(moving_tiles, rot_trans)}
        trans_x = {k : i for k, i in zip(moving_tiles, rot_trans[len(moving_tiles):])}
        trans_y = {k : i for k, i in zip(moving_tiles, rot_trans[2 * len(moving_tiles):])}
        rots[fixed_tile] = 0
        trans_x[fixed_tile] = 0
        trans_y[fixed_tile] = 0
        diffs = []
        for (k1, k2), matches in all_good_matches.iteritems():
            new_x_1, new_y_1 = transform_np(rots[k1], trans_x[k1], trans_y[k1], (matches[:, 0, 0], matches[:, 0, 1]))
            new_x_2, new_y_2 = transform_np(rots[k2], trans_x[k2], trans_y[k2], (matches[:, 1, 0], matches[:, 1, 1]))
            diffs = np.concatenate((diffs, np.sqrt((new_x_1 - new_x_2) ** 2 + (new_y_1 - new_y_2) ** 2)))
            
            for x1, y1, x2, y2 in zip(new_x_1, new_y_1, new_x_2, new_y_2):
                pylab.plot([y1, y2], [x1, x2], 'k', lw=0.5)

        pylab.title("L1 error: %0.3f\nmedian distance: %0.2f" % (err, np.median(diffs)))

        pylab.axis('off')
        if idx == 0:
            pylab.axis('equal')
            next_ax = pylab.axis()
            print "next_ax", next_ax
        else:
            pylab.axis(next_ax)
        pylab.savefig('frame_05_%05d' % idx, dpi=300)
        pylab.close()

    ppl.plot(intermediates)
    pylab.show()
    return best_w_b


if __name__ == '__main__':
    res = compute_alignments(sys.argv[1], sys.argv[2])

