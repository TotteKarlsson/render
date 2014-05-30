import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist

class Features(object):
    def __init__(self, locations, features):
        self.locations = locations
        self.features = features
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

    def match(self, other, max_difference=0.3, max_match_distance=1000):
        if len(set(self.bins.keys()) & set(other.bins.keys())) == 0:
            return ((self, []), (other, []))

        def get_near(F, x, y):
            return np.concatenate([F.bins[x + dx, y + dy] for dx in range(-1, 2) for dy in range(-1, 2)]).astype(int)

        def find_matches(F1, F1_indices, F2, F2_indices):
            features1 = F1.features[F1_indices, :]
            features2 = F2.features[F2_indices, :]
            if features1.size > 0 and features2.size > 0:
                diffs = cdist(features1, features2)
                mins = F2_indices[np.argmin(diffs, axis=1)]
                return mins, diffs.min(axis=1)
            else:
                return np.zeros(features1.size, int), np.ones(features1.size) * np.inf

        match_indices_1 = np.zeros(self.size, np.int)
        differences_1 = np.zeros(self.size)
        match_indices_2 = np.zeros(other.size, np.int)
        differences_2 = np.zeros(other.size)

        for binx, biny in set(self.bins.keys()) | set(other.bins.keys()):
            indices_1 = self.bins[binx, biny]
            indices_2 = other.bins[binx, biny]
            indices_1_extended = get_near(self, binx, biny)
            indices_2_extended = get_near(other, binx, biny)

            match_indices_1[indices_1], differences_1[indices_1] = find_matches(self, indices_1, other, indices_2_extended)
            match_indices_2[indices_2], differences_2[indices_2] = find_matches(other, indices_2, self, indices_1_extended)

        # only use unambiguous matches, that are close enough
        matches_1_index = np.arange(match_indices_1.size, dtype=np.int)
        unambiguous_1 = match_indices_2[match_indices_1] == matches_1_index
        close_enough_features = (differences_1 < max_difference)
        close_enough_distance = (np.linalg.norm(self.locations - other.locations[match_indices_1, :], axis=1) < max_match_distance)
        good_matches = matches_1_index[unambiguous_1 & close_enough_features & close_enough_distance]
        return good_matches, match_indices_1[good_matches]

    def update(self, other):
        self.locations = np.vstack((self.locations, other.locations))
        self.features = np.vstack((self.features, other.features))
        self.size = self.locations.shape[0]