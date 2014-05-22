import numpy as np

class Grid(object):
    def __init__(self, bbox, grid_size):
        self.bbox = bbox.copy()
        # build base grid
        bbshape = self.bbox.shape()

        self.num_steps_x = grid_size[0]
        self.num_steps_y = grid_size[1]
        self.stepsize_x = bbshape[0] / (self.num_steps_x - 1)
        self.stepsize_y = bbshape[1] / (self.num_steps_y - 1)

        x_grid, y_grid = np.mgrid[:self.num_steps_x, :self.num_steps_y].astype(np.float32)
        x_grid = self.stepsize_x * x_grid + self.bbox.from_x
        y_grid = self.stepsize_y * y_grid + self.bbox.from_y
        self.grid = np.hstack((x_grid.reshape((-1, 1)),
                               y_grid.reshape((-1, 1))))

    def weighted_combination(self, pts):
        assert np.all(self.bbox.contains(pts))
        assert pts.shape[1] == 2

        dx = pts[:, :1] - self.bbox.from_x
        dy = pts[:, 1:] - self.bbox.from_y
        xidx_lo = (dx / self.stepsize_x).astype(int)
        yidx_lo = (dy / self.stepsize_y).astype(int)
        w_x_lo = 1.0 - (dx - xidx_lo * self.stepsize_x) / self.stepsize_x
        w_y_lo = 1.0 - (dy - yidx_lo * self.stepsize_y) / self.stepsize_y
        assert w_x_lo.shape[0] == pts.shape[0]

        xidx_hi = np.minimum(xidx_lo + 1, self.num_steps_x - 1)
        yidx_hi = np.minimum(yidx_lo + 1, self.num_steps_y - 1)
        w_x_hi = 1.0 - w_x_lo
        w_y_hi = 1.0 - w_y_lo

        idx00 = np.ravel_multi_index((xidx_lo, yidx_lo), (self.num_steps_x, self.num_steps_y))
        idx01 = np.ravel_multi_index((xidx_lo, yidx_hi), (self.num_steps_x, self.num_steps_y))
        idx10 = np.ravel_multi_index((xidx_hi, yidx_lo), (self.num_steps_x, self.num_steps_y))
        idx11 = np.ravel_multi_index((xidx_hi, yidx_hi), (self.num_steps_x, self.num_steps_y))
        assert idx00.shape[0] == pts.shape[0]

        w00 = w_x_lo * w_y_lo
        w01 = w_x_lo * w_y_hi
        w10 = w_x_hi * w_y_lo
        w11 = w_x_hi * w_y_hi

        return (np.hstack((idx00, idx01, idx10, idx11)),
                np.hstack((w00, w01, w10, w11)))

    def structural(self):
        xidx, yidx = x_grid, y_grid = np.mgrid[:self.num_steps_x, :self.num_steps_y]
        lin_idx = np.ravel_multi_index((xidx, yidx), (self.num_steps_x, self.num_steps_y))
        indices1 = []
        indices2 = []
        lens = []

        indices1.append(lin_idx[1:, :].ravel())
        indices2.append(lin_idx[:-1, :].ravel())
        lens.append(self.stepsize_x * np.ones(indices1[-1].size))

        indices1.append(lin_idx[:, 1:].ravel())
        indices2.append(lin_idx[:, :-1].ravel())
        lens.append(self.stepsize_y * np.ones(indices1[-1].size))

        diag = np.hypot(self.stepsize_x, self.stepsize_y)

        indices1.append(lin_idx[1:, 1:].ravel())
        indices2.append(lin_idx[:-1, :-1].ravel())
        lens.append(diag * np.ones(indices1[-1].size))

        indices1.append(lin_idx[:-1, 1:].ravel())
        indices2.append(lin_idx[1:, :-1].ravel())
        lens.append(diag * np.ones(indices1[-1].size))

        indices1 = np.concatenate(indices1)
        indices2 = np.concatenate(indices2)
        lens = np.concatenate(lens)
        return indices1, indices2, lens
