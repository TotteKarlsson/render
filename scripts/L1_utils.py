import json
import numpy as np
from bounding_box import BoundingBox
from features import Features
from functools import reduce

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
    locations = np.vstack([np.array([f["location"] for f in features])]).reshape((-1, 2))
    if locations.size == 0:
        return Features(locations, np.empty((0, 0)))
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
        if f.size == 0:
            continue
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


