from theano import shared, function
import theano.tensor as T
import numpy as np

class Grid(object):
    def __init__(self, bbox, grid_size, offsets):
        self.bbox = bbox.copy()
        # build base grid
        bbshape = self.bbox.shape()

        self.num_steps_x = grid_size[0]
        self.num_steps_y = grid_size[1]
        self.stepsize_x = bbshape[0] / (self.num_steps_x - 1)
        self.stepsize_y = bbshape[1] / (self.num_steps_y - 1)

        self.grid_base = np.mgrid[:self.num_steps_x, :self.num_steps_y].astype(np.float32)
        self.grid_base[0, ...] *= self.stepsize_x
        self.grid_base[1, ...] *= self.stepsize_y
        self.grid_base += np.array([self.bbox.from_x, self.bbox.from_y]).reshape((2, 1, 1))
        self.offsets = offsets
        self.grid = shared(self.grid_base) + self.offsets
        assert self.grid_base.shape[0] == 2

        self.__end_positions_fun = function([self.offsets], self.grid, on_unused_input='ignore')

    def weighted_combination(self, pts, max_allowable_error=0.5):
        assert np.all(self.bbox.contains(pts))
        assert pts.shape[1] == 2

        dx = pts[:, 0] - self.bbox.from_x
        dy = pts[:, 1] - self.bbox.from_y
        xidx_lo = (dx / self.stepsize_x).astype(int)
        yidx_lo = (dy / self.stepsize_y).astype(int)
        w_x_lo = 1.0 - (dx - xidx_lo * self.stepsize_x) / self.stepsize_x
        w_y_lo = 1.0 - (dy - yidx_lo * self.stepsize_y) / self.stepsize_y
        assert w_x_lo.size == pts.shape[0]

        xidx_hi = np.minimum(xidx_lo + 1, self.num_steps_x - 1)
        yidx_hi = np.minimum(yidx_lo + 1, self.num_steps_y - 1)
        w_x_hi = 1.0 - w_x_lo
        w_y_hi = 1.0 - w_y_lo

        p00 = (w_x_lo * w_y_lo).reshape((1, -1)) * self.grid[:, xidx_lo, yidx_lo]
        p01 = (w_x_lo * w_y_hi).reshape((1, -1)) * self.grid[:, xidx_lo, yidx_hi]
        p10 = (w_x_hi * w_y_lo).reshape((1, -1)) * self.grid[:, xidx_hi, yidx_lo]
        p11 = (w_x_hi * w_y_hi).reshape((1, -1)) * self.grid[:, xidx_hi, yidx_hi]

        if max_allowable_error is not None:
            bp00 = (w_x_lo * w_y_lo).reshape((1, -1)) * self.grid_base[:, xidx_lo, yidx_lo]
            bp01 = (w_x_lo * w_y_hi).reshape((1, -1)) * self.grid_base[:, xidx_lo, yidx_hi]
            bp10 = (w_x_hi * w_y_lo).reshape((1, -1)) * self.grid_base[:, xidx_hi, yidx_lo]
            bp11 = (w_x_hi * w_y_hi).reshape((1, -1)) * self.grid_base[:, xidx_hi, yidx_hi]
            assert np.max(abs(pts - (bp00 + bp01 + bp10 + bp11).T)) < max_allowable_error

        return (p00 + p01 + p10 + p11).T

    def args(self):
        return self.offsets

    def args_size(self):
        return self.grid_base.size

    def args_shape(self):
        return self.grid_base.shape

    def start_end_positions(self, args):
        return self.grid_base, self.__end_positions_fun(args.reshape(self.grid_base.shape))

    def structural(self, smoothing=1):
        edge1 = self.grid[:, 1:, :] - self.grid[:, :-1, :]
        edge2 = self.grid[:, :, 1:] - self.grid[:, :, :-1]
        diag1 = self.grid[:, 1:, 1:] - self.grid[:, :-1, :-1]
        diag2 = self.grid[:, 1:, :-1] - self.grid[:, :-1, 1:]
        def delta_len(A, l):
            # smooth absolute value of difference
            return T.sum(T.sqrt(T.sqr(T.sqrt(T.sum(T.sqr(A), axis=0)) - l) + smoothing))
        return ((delta_len(edge1, self.stepsize_x) +
                 delta_len(edge2, self.stepsize_y) +
                 delta_len(diag1, np.sqrt(self.stepsize_x ** 2 + self.stepsize_y ** 2)) +
                 delta_len(diag2, np.sqrt(self.stepsize_x ** 2 + self.stepsize_y ** 2))) / 
                (self.num_steps_x * self.num_steps_y))
