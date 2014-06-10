import sys
import os.path
import os
import json
import cv2
import numpy as np

from L1_utils import load_tilespecs, load_features, load_transforms, save_transforms, extract_features, load_and_transform, rc, url2path


if __name__ == '__main__':
    rigid_transforms_file = sys.argv.pop()
    tile_files = sys.argv[1::2]
    input_transforms = sys.argv[2::2]

    assert len(tile_files) == len(input_transforms)

    tilenames = [os.path.basename(t).split('_')[1] for t in tile_files]

    rigid_transforms = json.load(open(rigid_transforms_file, "r"))

    video = None
    for tf, tn, trf in zip(tile_files, tilenames, input_transforms):
        tilespec = [ts for ts in load_tilespecs(tf) if rc(ts["mipmapLevels"]["0"]["imageUrl"]) == "r1-c1"][0]
        trans = [tr for tr in load_transforms(trf) if rc(tr["tile"]) == "r1-c1"][0]
        rigid_trans = rigid_transforms[tn]
        image_path = url2path(tilespec["mipmapLevels"]["0"]["imageUrl"])
        print "loading", image_path
        im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print "BAH"
            continue
        assert trans["rotation_rad"] == 0.0
        assert np.all(abs(np.array(trans["trans"])) < 1)
        R = rigid_trans["rotation_rad"]
        c = np.cos(R)
        s = np.sin(R)
        Tx, Ty = rigid_trans["trans"]
        M = np.array([[c, -s, Tx],
                      [s,  c, Ty]])
        print tn
        newim = cv2.warpAffine(im, M, im.shape[::-1])[::8, ::8]
        if video is None:
            height, width = newim.shape
            fourcc = cv2.cv.FOURCC(*'MJPG')
            video = cv2.VideoWriter('video.avi', fourcc, 20,(width, height))
        video.write(np.dstack([newim]*3))
video.release()
