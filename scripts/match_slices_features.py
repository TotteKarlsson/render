import sys
import json
from bounding_box import BoundingBox
import numpy as np
from scipy.spatial import cKDTree
import prettyplotlib as ppl

import theano.tensor as T
from theano import function, scan, shared
import scipy.optimize


alpha = 0.1

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

def extract_features(features):
    locations = np.vstack([np.array([f["location"] for f in features])])
    npfeatures = np.vstack([np.array([f["descriptor"] for f in features])])
    return locations, npfeatures

def offset_features(features, delta_x, delta_y):
    locations, _ = features
    locations[:, 0] += delta_x
    locations[:, 1] += delta_y
    return features

def transform_features(features, R, T):
    locations, _ = features

    s = np.sin(R)
    c = np.cos(R)

    tmp_x = locations[:, 0]
    tmp_y = locations[:, 1]
    new_x =   c * tmp_x + s * tmp_y + T[0]
    new_y = - s * tmp_x + c * tmp_y + T[1]
    locations[:, 0] = new_x
    locations[:, 1] = new_y

    return features


def load_and_transform_features(tilespec_file, feature_file, transform_file):
    bboxes = {ts["mipmapLevels"]["0"]["imageUrl"] : BoundingBox(*ts["bbox"]) for ts in load_tilespecs(tilespec_file)}
    features = {fs["mipmapLevels"]["0"]["imageUrl"] : fs["mipmapLevels"]["0"]["featureList"] for fs in load_features(feature_file)}
    transforms = {t["tile"] : (t["rotation_rad"], t["trans"]) for t in load_transforms(transform_file)}
    assert set(bboxes.keys()) == set(features.keys())
    assert set(bboxes.keys()) == set(transforms.keys())

    for k in bboxes:
        features[k] = extract_features(features[k])
        features[k] = offset_features(features[k], bboxes[k].from_x, bboxes[k].from_y)
        features[k] = transform_features(features[k], transforms[k][0], transforms[k][1])

    # build KDTrees

    # stack positions, data
    stacked_locations = np.vstack([l for l, f in features.values()])
    stacked_features = np.vstack([f for l, f in features.values()])
    print "building"
    return cKDTree(stacked_locations), stacked_features

def compute_alignment(features_1, features_2, max_distance=1500):
    print "about to match"
    tree1, features_1 = features_1
    tree2, features_2 = features_2

    spatial_distances = tree1.sparse_distance_matrix(tree2, max_distance)
    feature_distances = np.array([np.linalg.norm(features_1[idx1, :] - features_2[idx2, :]) for idx1, idx2 in spatial_distances.keys()])


    print "# matches below 0.3", (feature_distances < 0.3).sum(), "out of", feature_distances.size

    R = T.dscalar('rot')
    Tx = T.dscalar('Tx')
    Ty = T.dscalar('Ty')

    taken_1 = set()
    taken_2 = set()
    count = 0

    match_pts = []

    for d, (idx1, idx2) in sorted(zip(feature_distances, spatial_distances.keys())):
        if (d < 0.3) and (idx1 not in taken_1) and (idx2 not in taken_2):
            pt1 = tree1.data[idx1, :]
            pt2 = tree2.data[idx2, :]
            count += 1

            match_pts.append((pt1, pt2))

        taken_1.add(idx1)
        taken_2.add(idx2)


    def trans(px, py):
        s = T.sin(R / 360)
        c = T.cos(R / 360)
        new_x =   c * px + s * py + Tx
        new_y = - s * px + c * py + Ty
        return new_x, new_y


    pts = np.array(match_pts)
    pt1x = shared(pts[:, 0, 0])
    pt1y = shared(pts[:, 0, 1])
    pt2x = shared(pts[:, 1, 0])
    pt2y = shared(pts[:, 1, 1])
    pt2x, pt2y = trans(pt2x, pt2y)
    err = T.sum(T.sqrt(T.sqr(pt1x - pt2x) + T.sqr(pt1y - pt2y)))

    print "good matches", count

    err = err / count

    args = [R, Tx, Ty]
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
        if intermediates:
            print intermediates[-1][0]
        pass

    best_w_b = scipy.optimize.fmin_bfgs(f=f,
                                      x0=np.zeros(len(args)),
                                      fprime=g,
                                      callback=callback,
                                      maxiter=5000,
                                      disp=1)

    best_w_b[0] /= 360
    return best_w_b

if __name__ == '__main__':
    tile_file_1, features_file_1, transforms_file_1 = sys.argv[1:4]
    tile_file_2, features_file_2, transforms_file_2, new_transforms = sys.argv[4:]
    updated_transforms = load_transforms(transforms_file_2)

    features_1 = load_and_transform_features(tile_file_1, features_file_1, transforms_file_1)
    print "done"
    features_2 = load_and_transform_features(tile_file_2, features_file_2, transforms_file_2)
    print "done"
    R, Tx, Ty = compute_alignment(features_1, features_2)

    for t in updated_transforms:
        s = np.sin(R)
        c = np.cos(R)
        t['rotation_rad'] += R
        tx, ty = t['trans']
        new_x =   c * tx + s * ty + Tx
        new_y = - s * tx + c * ty + Ty
        t['trans'] = [new_x, new_y]

    with open(new_transforms, "wb") as output:
        json.dump(updated_transforms, output,
                  sort_keys=True,
                  indent=4)
