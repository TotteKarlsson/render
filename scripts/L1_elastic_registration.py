import sys
import json
import numpy as np
import itertools
from functools import reduce
import pylab as plt

from hesfree import hessian_free

import scipy.optimize
import scipy.sparse as sp

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


def match_distances(features_1, features_2, max_distance=1500, fthresh=0.25):
    features_1.bin(max_distance)
    features_2.bin(max_distance)

    idx1, idx2 = features_1.match(features_2, max_difference=fthresh, max_match_distance=max_distance)
    orig_dists = np.sqrt(np.sum((features_1.locations[idx1, :] - features_2.locations[idx2, :]) ** 2, axis=1))
    return idx1, idx2, orig_dists

def weight(skip):
    return 1.25 ** -(abs(skip) - 1)


if __name__ == '__main__':
    if False:  # truncate
        sys.argv = sys.argv[:13]

    tile_files = sys.argv[1::4]
    feature_files = sys.argv[2::4]
    montage_transforms = sys.argv[3::4]
    input_transforms = sys.argv[4::4]

    print len(tile_files), "tiles to be processed"

    grid_size = (20, 30)
    stiffness = 1.0

    transformed_bboxes, transformed_features = zip(*[load_and_transform(tf, ff, it)
                                                     for tf, ff, it in
                                                     zip(tile_files, feature_files, input_transforms)])
    grids = [Grid(t, grid_size) for t in transformed_bboxes]
    num_slices = len(grids)

    grid_offsets = np.prod(grid_size) * np.arange(len(grids), dtype=int)
    feature_offsets = [0] + np.cumsum([f.size for f in transformed_features]).tolist()
    num_features = np.sum([f.size for f in transformed_features])

    # W takes grid points to feature points
    W_shape = (np.sum(num_features), len(grids) * np.prod(grid_size))
    W = sp.csr_matrix(W_shape)
    for g, f, go, fo in zip(grids, transformed_features, grid_offsets, feature_offsets):
        grid_idx, weights = g.weighted_combination(f.locations)
        pt_idx = np.hstack([np.arange(weights.shape[0], dtype=int).reshape((-1, 1))] * 4)
        W = W + sp.csr_matrix((weights.ravel(), (fo + pt_idx.ravel(), go + grid_idx.ravel())), shape=W_shape)
    in_grid = np.vstack([g.grid for g in grids])
    outpts = W * in_grid

    # test reconstruction error
    for f, fo in zip(transformed_features, feature_offsets):
        sub = outpts[fo:fo+f.locations.shape[0], :]
        assert np.max(np.abs(f.locations - sub)) < 0.01

    # build links between features
    print "linking"
    total_matches = 0
    link_idx1 = []
    link_idx2 = []
    desired_distances = []
    weights = []
    orig_dists = []
    for sl_idx1, sl_idx2 in itertools.combinations(range(num_slices), 2):
        if abs(sl_idx1 - sl_idx2) <= 5:
            pt_idx1, pt_idx2, curdists = match_distances(transformed_features[sl_idx1], transformed_features[sl_idx2])
            orig_dists.append(curdists)
            link_idx1.append(pt_idx1 + feature_offsets[sl_idx1])
            link_idx2.append(pt_idx2 + feature_offsets[sl_idx2])
            total_matches += pt_idx1.size
            desired_distances.append(np.zeros(pt_idx1.size))
            weights.append(weight(sl_idx1 - sl_idx2) * np.ones(pt_idx1.size))

    # build structural links
    print "structure"
    structural_W = sp.eye(len(grids) * np.prod(grid_size))  # weight matrix for grid points taken to themselves
    structure_links = [g.structural() for g in grids]
    # all structural points will be after feature points
    for (idx1, idx2, _), go in zip(structure_links, grid_offsets):
        idx1 += go + W_shape[0]
        idx2 += go + W_shape[0]

    W = sp.vstack((W, structural_W))

    # build point selection matrices
    structural_indices_1, structural_indices_2, structural_dists = zip(*structure_links)
    indices_a = np.concatenate([idx.ravel() for idx in link_idx1] + [idx.ravel() for idx in structural_indices_1])
    indices_b = np.concatenate([idx.ravel() for idx in link_idx2] + [idx.ravel() for idx in structural_indices_2])

    num_links = indices_a.size
    Pa = sp.csr_matrix((np.ones(num_links), (range(num_links), indices_a)), shape=(num_links, W.shape[0]))
    Pb = sp.csr_matrix((np.ones(num_links), (range(num_links), indices_b)), shape=(num_links, W.shape[0]))

    # M takes grids to deltas
    M = (Pa - Pb) * W

    deltas = M * in_grid
    deltas = np.sqrt(np.sum(deltas ** 2, axis=1))
    orig_dists = np.concatenate(orig_dists)
    structural_dists = np.concatenate(structural_dists)
    Dpts = deltas[:orig_dists.size]
    print "diff between linked", np.max(np.abs(orig_dists - Dpts))
    Spts = deltas[orig_dists.size:]
    print "diff between structural", np.max(np.abs(structural_dists - Spts))


    weights_links = np.concatenate(weights) / total_matches
    num_structural = sum(s.size for s in structural_indices_1)
    weight_structural = stiffness * np.ones(num_structural) / num_structural

    desired_lengths = np.concatenate([np.zeros(total_matches), structural_dists]).reshape((-1, 1))
    # add a smoother
    desired_lengths = np.sqrt(desired_lengths**2 + 1)
    weights = np.concatenate([weights_links, weight_structural]).reshape((-1, 1))

    def err(offset):
        AB = M * (in_grid + offset.reshape(in_grid.shape))
        lens = np.sqrt((AB ** 2).sum(axis=1) + 1).reshape((-1, 1))
        errs = np.sqrt((lens - desired_lengths)**2 + 1)
        return np.sum(errs * weights) - np.sum(weights)  # minimum at 0

    def err_gradient(offset):
        AB = M * (in_grid + offset.reshape(in_grid.shape))
        lens = np.sqrt((AB ** 2).sum(axis=1) + 1).reshape((-1, 1))
        delta_lens = (lens - desired_lengths)
        dAB = (AB / lens) * (delta_lens / np.sqrt(delta_lens**2 + 1))
        gXY = M.T * (dAB * weights)
        return gXY.ravel()


    def err_hessp(offset, v):
        # compute as d(gradient(G + e * v)) / de at e == 0
        AB = M * (in_grid + offset.reshape(in_grid.shape))
        Mv = M * v.reshape(in_grid.shape)
        # dAB / de = Mv

        lens = np.sqrt((AB ** 2).sum(axis = 1) + 1).reshape((-1, 1))
        normed_AB = AB / lens

        # d [sqrt(x**2 + y**2  + 1)] / dx = x / sqrt(x**2 + y**2  + 1)
        d_lens_de = (Mv * normed_AB).sum(axis=1).reshape((-1, 1))
        # quotient rule
        d_NAB_de = ((lens * Mv) - (AB * d_lens_de)) / lens**2

        delta_lens = (lens - desired_lengths)
        normed_delta_lens = delta_lens / np.sqrt(delta_lens**2 + 1)
        # d delta_lens / de = d lens / de
        # d normed_delta_lens / de = (1 / (np.sqrt(delta_lens**2 + 1))**3) * (d delta_lens / de)
        d_NDL_de = d_lens_de / np.sqrt(delta_lens**2 + 1)**3

        d_AB_de = normed_AB * (d_NDL_de) + normed_delta_lens * d_NAB_de
        return (M.T * (d_AB_de * weights)).ravel()


    if False:
        print "err"
        err0 = err(np.zeros(in_grid.size))
        print "ERR", err0
        gr = err_gradient(np.zeros(in_grid.size))
        sp = np.zeros(in_grid.size)
        dd = np.zeros(in_grid.size)
        for idx in range(in_grid.size):
            sp[idx] = 0.0000001
            dd[idx] = (err(sp) - err(-sp)) / (2 * 0.0000001)
            sp[idx] = 0
        print "max", np.max(abs(dd - gr)), np.median(abs(dd))

    def callback(x):
        E = err(x)
        gnorm = np.linalg.norm(err_gradient(x))
        AB = M * (in_grid + x.reshape(in_grid.shape))
        lens = np.sqrt((AB ** 2).sum(axis=1)).reshape((-1, 1))
        med = np.median(lens[:total_matches])
        print "error: %0.2f     median_dist: %0.2f  gnorm: %0.4f" % (E, med, gnorm),
        print np.linalg.norm(x)

    best_offset = hessian_free(f=err,
                               x0=np.zeros(in_grid.size),
                               fprime=err_gradient,
                               fhessp=err_hessp,
                               callback=callback,
                               maxiter=100)


import pylab
out_grid = in_grid + best_offset.reshape(in_grid.shape)
grid3 = out_grid[grid_offsets[3]:grid_offsets[4], :]
pylab.plot(grid3[:, 1], grid3[:, 0], '.')
pylab.figure()
base = in_grid[grid_offsets[3]:grid_offsets[4], :]
offsets = best_offset.reshape(in_grid.shape)[grid_offsets[3]:grid_offsets[4], :]
pylab.quiver(base[:, 1], base[:, 0], offsets[:, 1], offsets[:, 0])
pylab.show()

