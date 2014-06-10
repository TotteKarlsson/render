import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
import timer

import pyximport; pyximport.install()
from match_features import match_features

class Features(object):
    def __init__(self, locations, features):
        self.locations = locations.astype(np.float32)
        self.features = features.astype(np.float32)
        self.size = self.locations.shape[0]
        self.bins = None
        self.bin_radius = None

    def offset(self, delta_x, delta_y):
        self.locations[:, 0] += delta_x
        self.locations[:, 1] += delta_y

    def transform(self, R, T):
        s = np.sin(R)
        c = np.cos(R)
        tmp_x = self.locations[:, 0]
        tmp_y = self.locations[:, 1]
        new_x =  c * tmp_x - s * tmp_y + T[0]
        new_y =  s * tmp_x + c * tmp_y + T[1]
        self.locations[:, 0] = new_x
        self.locations[:, 1] = new_y

    def bin(self, radius):
        if self.bin_radius == radius:
            return
        self.bin_radius = radius
        quantized = (self.locations / radius).astype(int)
        max_x, max_y = np.max(quantized, axis=0)
        min_x, min_y = np.min(quantized, axis=0)
        self.bins = defaultdict(list)
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                self.bins[x, y] = np.nonzero((quantized[:, 0] == x) & (quantized[:, 1] == y))[0]

    def match(self, other, max_difference=0.45, max_match_distance=3500):
        match_1, match_2, diffs = match_features(self.locations, self.features,
                                                 other.locations, other.features,
                                                 max_match_distance)
        mask = (match_2[match_1] == np.arange(match_1.size)) & (diffs < max_difference)
        set1 = np.nonzero(mask)[0]
        set2 = match_1[mask]
        diffs = diffs[mask]

        dists = np.linalg.norm(self.locations[set1, :] - other.locations[set2, :], axis=1)
        print "    found", mask.sum(), "MED", np.median(dists), "MAD", np.median(abs(dists - np.median(dists)))
        return set1, set2, diffs, dists

    def update(self, other):
        self.locations = np.vstack((self.locations, other.locations)).astype(np.float32)
        self.features = np.vstack((self.features, other.features)).astype(np.float32)
        self.size = self.locations.shape[0]
