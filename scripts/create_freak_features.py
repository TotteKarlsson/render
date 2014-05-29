import cv2
import sys
import numpy as np
import argparse
import os
import os.path
import glob
import ujson as json
from timer import Timer

import urllib
try:   
    from urlparse import urljoin
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urljoin
    from urllib.parse import urlparse


def url2path(url):
    p = urlparse(url)
    return os.path.join(p.netloc, p.path)


def tilegen(image, tile_size=1024, overlap=0):
    for i in range(0, image.shape[0], tile_size - overlap):
        for j in range(0, image.shape[1], tile_size - overlap):
            yield i, j, image[i:(i + tile_size),
                              j:(j + tile_size)]


threshold = 20
def compute_features(image_path, desired_count=20000):
    im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(im, (0, 0), 4.0, im)
    im = im[::2, ::2].copy()

    global threshold
    while True:
        fast = cv2.FastFeatureDetector(threshold=threshold, nonmaxSuppression=True)
        pts = fast.detect(im)
        if (desired_count / 1.5) > len(pts):
            print "too few", len(pts), "moved thresh to", threshold
            treshold -= 1
        elif len(pts) > (1.5 * desired_count):
            print "too many", len(pts), "moved thresh to", threshold
            threshold += 1
        else:
            break
        fast.setDouble('threshold', threshold)

    print len(pts), "points"

    freak = cv2.DescriptorExtractor_create('FREAK')
    pts, descriptors = freak.compute(im, pts)
    descriptors = descriptors.view(np.int64)

    return [{"location" : (np.array(k.pt) * 2).tolist(),
             "descriptor" : d.tolist()} for k, d in zip(pts, descriptors)]

def compute_features_all_tiles(tile_file, out_dir):
    fname, ext = os.path.splitext(os.path.basename(tile_file))
    sift_out_file = os.path.join(out_dir, '{0}_freakFeatures.json'.format(fname))

    output_list = []
    with open(tile_file) as f:
        tilespecs = json.load(f)
        for tilespec in tilespecs:
            filename = tilespec["mipmapLevels"]["0"]["imageUrl"]
            with Timer("compute features {}".format(filename)):
                image_path = url2path(filename)
                feature_list = compute_features(image_path)
                tile_info = {"mipmapLevels" :
                                 {"0" : {"imageUrl" : filename,
                                         "featureList": feature_list}}}
                output_list.append(tile_info)

        if len(output_list) > 0:
            with open(sift_out_file, 'w') as outfile:
                json.dump(output_list, outfile, ensure_ascii=False)


def create_FREAK_features(tiles_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tile_files = glob.glob(os.path.join(tiles_dir, '*.json'))
    for tile_file in tile_files:
        compute_features_all_tiles(tile_file, output_dir)

def main():
    # Command line parser
    parser = argparse.ArgumentParser(description='Iterates over a directory that contains json files, \
        and creates the FREAK features of each file. \
        The output is either in the same directory or in a different, user-provided, directory \
        (in either case, we use a different file name).')
    parser.add_argument('tiles_dir', metavar='tiles_dir', type=str, 
                        help='a directory that contains tile_spec files. FREAK features will be extracted from each tile')
    parser.add_argument('-o', '--output_dir', type=str, 
                        help='a directory where the output files will be kept (default: ./temp)',
                        default='./temp')
    args = parser.parse_args()

    create_FREAK_features(args.tiles_dir, args.output_dir)

if __name__ == '__main__':
    main()
