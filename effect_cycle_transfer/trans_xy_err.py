import numpy as np


class error_rec(object):
    def __init__(self, x_arg, y_arg):
        self.x_arg = x_arg
        self.y_arg = y_arg
        self.count = 0
        self.err_mean = 0.
        self.err_var = 0.
        self.err_max = 0.

    def __call__(self, ori_state, trans_state):
        ori_xy = np.array([ori_state[self.x_arg], ori_state[self.y_arg]])
        trans_xy = np.array([trans_state[self.x_arg], trans_state[self.y_arg]])
        error = np.linalg.norm(ori_xy - trans_xy)
        if error > self.err_max:
            self.err_max = error

        err_mean_next = (self.err_mean * self.count + error) / (self.count + 1)

        self.err_var = (self.count * (
                self.err_var + self.err_mean ** 2 - err_mean_next ** 2) + error ** 2 - err_mean_next ** 2) / (
                               self.count + 1)
        self.err_mean = err_mean_next
        self.count += 1

    def reset(self):
        self.count = 0
        self.err_mean = 0.
        self.err_var = 0.
        self.err_max = 0.

    @property
    def mean(self):
        return self.err_mean

    @property
    def var(self):
        return self.err_var
