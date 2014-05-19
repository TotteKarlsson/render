import sys
import json
import numpy as np
import itertools
from functools import reduce
import pylab as plt

import theano.tensor as T
from theano import function, scan, shared, pp, printing
import scipy.optimize

from bounding_box import BoundingBox
from features import Features
from grid import Grid

eps = np.finfo(np.float32).eps

def rc(filename):
    return filename.split('/')[-1][5:][:5]

def load_tilespecs(file):
    with open(file) as fp:
        return json.load(fp)

def load_features(file):
    with open(file) as fp:
        return json.load(fp)

def load_transforms(file):
    with open(file) as fp:
        return json.load(fp)

def save_transforms(file, trans):
    with open(file, "wb") as fp:
        json.dump(trans, fp, sort_keys=True, indent=4)

def extract_features(features):
    locations = np.vstack([np.array([f["location"] for f in features])])
    npfeatures = np.vstack([np.array([f["descriptor"] for f in features])])
    return Features(locations, npfeatures)

def load_and_transform(tilespec_file, feature_file, transform_file):
    bboxes = {ts["mipmapLevels"]["0"]["imageUrl"] : BoundingBox(*ts["bbox"]) for ts in load_tilespecs(tilespec_file)}
    features = {fs["mipmapLevels"]["0"]["imageUrl"] : fs["mipmapLevels"]["0"]["featureList"] for fs in load_features(feature_file)}
    transforms = {t["tile"] : (t["rotation_rad"], t["trans"]) for t in load_transforms(transform_file)}
    assert set(bboxes.keys()) == set(features.keys())
    assert set(bboxes.keys()) == set(transforms.keys())

    # offset and transform feature points
    all_features = None
    for k in bboxes:
        f = extract_features(features[k])
        f.offset(bboxes[k].from_x, bboxes[k].from_y)
        f.transform(transforms[k][0], transforms[k][1])
        if all_features:
            all_features.update(f)
        else:
            all_features = f

    # find union of transformed bounding boxes
    for tilename in bboxes.keys():
        R, T = transforms[tilename]
        bboxes[tilename] = bboxes[tilename].transform(R, T)
    full_bbox = reduce(lambda x, y: x.union(y), bboxes.values())

    assert np.all(full_bbox.contains(all_features.locations))
    return full_bbox, all_features


def match_distances(features_1, features_2, grid_1, grid_2, max_distance=1500, fthresh=0.25):
    features_1.bin(max_distance)
    features_2.bin(max_distance)

    matches = features_1.match(features_2, max_difference=fthresh, max_match_distance=max_distance)
    (_, idx1), (_, idx2) = matches
    if len(idx1) == 0:
        return [], 0

    ptdists = []
    warped_1 = grid_1.weighted_combination(features_1.locations[idx1, :])
    warped_2 = grid_2.weighted_combination(features_2.locations[idx2, :])
    

    orig_dists = np.sqrt(np.sum((features_1.locations[idx1, :] - features_2.locations[idx2, :]) ** 2, axis=1))

    # assume pts come back as Nx2 arrays
    # add eps to make perfect matches differentiable
    return T.sqrt(T.sum(T.sqr(warped_1 - warped_2), axis=1) + eps), len(idx1), orig_dists


def weight(skip):
    return 1.25 ** -(abs(skip) - 1)

def make_function(*args):
    return function(*args, on_unused_input='ignore')


if __name__ == '__main__':
    tile_files = sys.argv[1::4]
    feature_files = sys.argv[2::4]
    montage_transforms = sys.argv[3::4]
    input_transforms = sys.argv[4::4]
    grid_size = (20, 30)
    stiffness = 1.0

    num_slices = len(tile_files)
    offsets_shape = [num_slices, 2, grid_size[0], grid_size[1]]
    params = T.vector('params')
    offsets = params.reshape(offsets_shape)

    transformed_bboxes, transformed_features = zip(*[load_and_transform(tf, ff, it)
                                                     for tf, ff, it in
                                                     zip(tile_files, feature_files, input_transforms)])
    slnone = slice(None, None, None)
    grids = [Grid(t, grid_size, offsets[idx, :, :, :])
             for idx, t in enumerate(transformed_bboxes)]

    dists = []
    orig_dists = []
    error_terms = []
    total_matches = 0
    for idx1, idx2 in itertools.combinations(range(num_slices), 2):
        curdists, match_count, orig = match_distances(transformed_features[idx1], transformed_features[idx2],
                                                grids[idx1], grids[idx2])
        total_matches += match_count
        print "    matching", idx1, idx2, match_count
        dists.append(curdists)
        orig_dists.append(orig)
        error_terms.append(T.sum(weight(idx1 - idx2) * T.sum(curdists)))

    dists = T.concatenate(dists)
    orig_dists = np.concatenate(orig_dists)
    error = T.sum(error_terms) / total_matches
    st = stiffness * T.sum([T.sum(g.structural()) for g in grids]) / len(grids)
    error = error + st

    v = T.vector('v')
    errgrad = T.grad(error, params)
    vH = T.grad(T.sum(errgrad * v), params)

    distfun = make_function([params], dists)
    errfun = make_function([params], error)
    errgradfun = make_function([params], errgrad)
    errhesspfun = make_function([params, v], vH)
    posfun = make_function([params], grids[3].grid)

    params_size = np.prod(offsets_shape)

    print "START", np.median(orig_dists), np.median(distfun(np.zeros(params_size, np.float32)))


    plt.ion()
    plt.figure(figsize=(8, 5))
    plt.show()

    def f(x):
        return errfun(x.astype(np.float32))

    def g(x):
        return errgradfun(x.astype(np.float32))

    def hv(x, v):
        return errhesspfun(x.astype(np.float32), v.astype(np.float32))

    def callback(x):
        err = errfun(x.astype(np.float32))
        gnorm = np.linalg.norm(errgradfun(x.astype(np.float32)))
        pos = posfun(x.astype(np.float32))
        xpos = pos[0, :, :].ravel()
        ypos = pos[1, :, :].ravel()
        width = np.ptp(xpos)
        height = np.ptp(ypos)
        med = np.median(distfun(x.astype(np.float32)))
        print "error: %0.2f     median_dist: %0.2f  gnorm: %0.4f  wh: %0.2f %0.2f %0.2f" % \
            (err, med, gnorm, width, height, width*height)
        plt.cla()
        plt.plot(ypos, xpos, '.')
        plt.axis('tight')
        plt.draw()

best_w_b = scipy.optimize.fmin_ncg(f=f,
                                      x0=np.zeros(params_size),
                                   fprime=g,
                                   fhess_p=hv,
                                      callback=callback,
                                      maxiter=5000,
                                      disp=False)
