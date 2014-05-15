import sys
import json
from bounding_box import BoundingBox
import numpy as np
from features import Features
import itertools

import theano.tensor as T
from theano import function, scan, shared
import scipy.optimize

from transform import Transform

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

def compose_transformations(A, B):
    rA = A["rotation_rad"]
    txA, tyA = A["trans"]
    rB = B["rotation_rad"]
    txB, tyB = B["trans"]

    s = np.sin(rA)
    c = np.cos(rA)

    new_tx = txA + c * txB - s * tyB
    new_ty = tyA + s * txB + c * tyB

    return {"tile" : A.get("tile", None) or B["tile"],
            "rotation_rad" : rA + rB,
            "trans" : [new_tx, new_ty]}

def save_transforms(file, trans):
    with open(file, "wb") as fp:
        json.dump(trans, fp, sort_keys=True, indent=4)

def extract_features(features):
    locations = np.vstack([np.array([f["location"] for f in features])])
    npfeatures = np.vstack([np.array([f["descriptor"] for f in features])])
    return Features(locations, npfeatures)

def load_and_transform_features(tilespec_file, feature_file, transform_file):
    bboxes = {ts["mipmapLevels"]["0"]["imageUrl"] : BoundingBox(*ts["bbox"]) for ts in load_tilespecs(tilespec_file)}
    features = {fs["mipmapLevels"]["0"]["imageUrl"] : fs["mipmapLevels"]["0"]["featureList"] for fs in load_features(feature_file)}
    transforms = {t["tile"] : (t["rotation_rad"], t["trans"]) for t in load_transforms(transform_file)}
    assert set(bboxes.keys()) == set(features.keys())
    assert set(bboxes.keys()) == set(transforms.keys())

    all_features = None
    for k in bboxes:
        f = extract_features(features[k])
        f.offset(bboxes[k].from_x, bboxes[k].from_y)
        f.transform(transforms[k][0], transforms[k][1])
        if all_features:
            all_features.update(f)
        else:
            all_features = f

    return all_features


def match_distances(features_1, features_2, trans_1, trans_2, max_distance=1500, fthresh=0.25):
    features_1.bin(max_distance)
    features_2.bin(max_distance)

    matches = features_1.match(features_2, max_difference=fthresh, max_match_distance=max_distance)
    (_, idx1), (_, idx2) = matches
    if len(idx1) == 0:
        return [], 0

    p1 = trans_1.transform(features_1.locations[idx1, :])
    p2 = trans_2.transform(features_2.locations[idx2, :])
    dists = T.sqrt(T.sum(T.sqr(p1 - p2), axis=1) + eps)  # eps to make perfect matches differentiable
    return dists, len(idx1)

def weight(skip):
    return 1.25 ** -(abs(skip) - 1)


if __name__ == '__main__':
    tile_files = sys.argv[1::4]
    feature_files = sys.argv[2::4]
    input_transforms = sys.argv[3::4]
    output_transforms = sys.argv[4::4]

    feature_trees = [load_and_transform_features(tf, ff, it)
                     for tf, ff, it in
                     zip(tile_files, feature_files, input_transforms)]

    transforms = [Transform(name=tf, fixed=(tf == tile_files[0])) for tf in tile_files]

    dists = []
    error_terms = []
    total_matches = 0
    for idx1, idx2 in itertools.combinations(range(len(feature_trees)), 2):
        curdists, match_count = match_distances(feature_trees[idx1], feature_trees[idx2],
                                                transforms[idx1], transforms[idx2])
        total_matches += match_count
        print "    matching", idx1, idx2, match_count
        dists.append(curdists)
        error_terms.append(T.sum(weight(idx1 - idx2) * T.sum(curdists)))

    args = [arg for t in transforms for arg in t.args()]
    dists = T.concatenate(dists)
    error = T.sum(error_terms) / total_matches


    distfun = function(args, dists)
    errfun = function(args, error)
    errgrad = function(args, T.grad(error, args))

    def f(x):
        result = errfun(*(x.tolist()))
        return result

    def g(x):
        result = np.array(errgrad(*(x.tolist())))
        return result

    def callback(x):
        print "error: %0.2f     median_dist: %0.2f" % (errfun(*x.astype(np.float32)),
                                                       np.median(distfun(*x.astype(np.float32))))

    best_w_b = scipy.optimize.fmin_bfgs(f=f,
                                        x0=np.zeros(len(args)),
                                        fprime=g,
                                        callback=callback,
                                        maxiter=5000,
                                        disp=False)
    updates = best_w_b.tolist()
    # write updated transforms
    for in_trans_f, out_trans_f, update_trans in zip(input_transforms, output_transforms, transforms):
        in_transformations = load_transforms(in_trans_f)
        num_args = len(update_trans.args())
        update = update_trans.args_to_dict(updates[:num_args])
        updates = updates[num_args:]
        new_transformations = [compose_transformations(update, t) for t in in_transformations]
        save_transforms(out_trans_f, new_transformations)
