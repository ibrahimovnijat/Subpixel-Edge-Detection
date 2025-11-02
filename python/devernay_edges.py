import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
import time


def dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))


def greater(a, b):
    if a <= b:
        return False
    if (a - b) < 1e-7:
        return False
    return True


class DevernayEdges:
    """Class for performing Devernay edge detection on an image.
    """
    def __init__(self, image, sigma=0.0, high_thresh=0.0, low_thresh=0.0):
        self.sigma = sigma
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.image = image

        self.image = self.image.convert("L")  # convert to grayscale
        [self.img_x, self.img_y] = self.image.size
        self.image_vec = np.ravel(np.array(self.image,
                                           dtype=np.float64)).reshape(
                                               [self.img_x * self.img_y, 1])

        self.Gx = np.zeros([self.img_x * self.img_y, 1], dtype=np.float64)
        self.Gy = np.zeros([self.img_x * self.img_y, 1], dtype=np.float64)
        self.modG = np.zeros([self.img_x * self.img_y, 1], dtype=np.float64)

        self.Ex = -np.ones((self.img_x * self.img_y, 1), dtype=np.float64)
        self.Ey = -np.ones((self.img_x * self.img_y, 1), dtype=np.float64)

        self.next_ = -np.ones((self.img_x * self.img_y, 1), dtype=np.int32)
        self.prev_ = -np.ones((self.img_x * self.img_y, 1), dtype=np.int32)

        self.edges_x = []
        self.edges_y = []

    def compute_gradient(self):
        for x in range(1, self.img_x - 1):
            for y in range(1, self.img_y - 1):
                self.Gx[x + (y * self.img_x)] = self.image_vec[
                    (x + 1) +
                    (y * self.img_x)] - self.image_vec[(x - 1) +
                                                       (y * self.img_x)]
                self.Gy[x + (y * self.img_x)] = self.image_vec[x + (
                    (y + 1) * self.img_x)] - self.image_vec[x + (
                        (y - 1) * self.img_x)]
                self.modG[x + (y * self.img_x)] = np.sqrt(
                    self.Gx[x + (y * self.img_x)] * self.Gx[x +
                                                            (y * self.img_x)] +
                    self.Gy[x + (y * self.img_x)] * self.Gy[x +
                                                            (y * self.img_x)])

    def list_chained_edge_points(self):
        for i in range(0, self.img_x * self.img_y):
            if self.prev_[i] >= 0 or self.next_[i] >= 0:
                k = i
                n = np.squeeze(self.prev_[k])
                while n >= 0 and n != i:
                    k = n
                    n = np.squeeze(self.prev_[k])

                while True:
                    self.edges_x.append(np.squeeze(self.Ex[k]))
                    self.edges_y.append(np.squeeze(self.Ey[k]))

                    n = np.squeeze(self.next_[k])
                    self.next_[k] = -1
                    self.prev_[k] = -1
                    k = n
                    if k < 0:
                        break

    def chain_edge_points(self):
        for x in range(2, self.img_x - 2):
            for y in range(2, self.img_y - 2):
                if self.Ex[x + (y * self.img_x)] >= 0.0 and self.Ey[
                        x + (y * self.img_x)] >= 0.0:
                    from_ = np.int32(x + y * self.img_x)
                    fwd_s = 0.0
                    bck_s = 0.0
                    fwd = -1
                    bck = -1

                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            to_ = np.int32(x + i + (y + j) * self.img_x)
                            s = self.chain(from_, to_)

                            if s > fwd_s:
                                fwd_s = s
                                fwd = to_

                            if s < bck_s:
                                bck_s = s
                                bck = to_

                    if fwd >= 0 and self.next_[from_] != fwd:
                        alt = self.prev_[fwd]
                        if alt < 0 or self.chain(alt, fwd) < fwd_s:
                            if self.next_[from_] >= 0:
                                self.prev_[self.next_[from_]] = -1
                            self.next_[from_] = fwd

                            if alt >= 0:
                                self.next_[alt] = -1
                            self.prev_[fwd] = from_

                    if bck >= 0 and self.prev_[from_] != bck:
                        alt = self.next_[bck]
                        if alt < 0 or self.chain(alt, bck) > bck_s:
                            if alt >= 0:
                                self.prev_[alt] = -1
                            self.next_[bck] = from_

                            if self.prev_[from_] >= 0:
                                self.next_[self.prev_[from_]] = -1
                            self.prev_[from_] = bck

    def thresholds_with_hysteresis(self):
        valid = np.full((self.img_x * self.img_y, 1), False)

        for i in range(0, self.img_x * self.img_y):
            if (self.prev_[i] >= 0 or self.next_[i] >= 0) and (
                    not valid[i]) and (self.modG[i] >= self.high_thresh):
                valid[i] = True

                j = i
                k = self.next_[j]
                while j >= 0 and k >= 0 and (not valid[k]):
                    if self.modG[k] < self.low_thresh:
                        self.next_[j] = -1
                        self.prev_[k] = -1
                    else:
                        valid[k] = True

                    j = np.squeeze(self.next_[j])
                    k = np.squeeze(self.next_[j])

                j = i
                k = self.prev_[j]
                while j >= 0 and k >= 0 and (not valid[k]):
                    if self.modG[k] < self.low_thresh:
                        self.prev_[j] = -1
                        self.prev_[k] = -1
                    else:
                        valid[k] = True

                    j = np.squeeze(self.prev_[j])
                    k = np.squeeze(self.prev_[j])

        for i in range(0, self.img_x * self.img_y):
            if (self.prev_[i] >= 0 or self.next_[i] >= 0) and not valid[i]:
                self.prev_[i] = -1
                self.next_[i] = -1

    def compute_edge_points(self):
        for x in range(2, self.img_x - 2):
            for y in range(2, self.img_y - 2):
                Dx = 0
                Dy = 0

                mod = self.modG[x + (y * self.img_x)]
                L = self.modG[x - 1 + (y * self.img_x)]
                R = self.modG[x + 1 + (y * self.img_x)]
                U = self.modG[x + ((y + 1) * self.img_x)]
                D = self.modG[x + ((y - 1) * self.img_x)]
                gx = np.fabs(self.Gx[x + (y * self.img_x)])
                gy = np.fabs(self.Gy[x + (y * self.img_x)])

                if greater(mod, L) and (not greater(R, mod)) and gx >= gy:
                    Dx = 1
                elif greater(mod, D) and (not greater(U, mod)) and gx <= gy:
                    Dy = 1

                if Dx > 0 or Dy > 0:
                    a = self.modG[x - Dx + ((y - Dy) * self.img_x)]
                    b = self.modG[x + (y * self.img_x)]
                    c = self.modG[x + Dx + ((y + Dy) * self.img_x)]
                    offset = 0.5 * (a - c) / (a - b - b + c)

                    self.Ex[x + (y * self.img_x)] = x + (offset * Dx)
                    self.Ey[x + (y * self.img_x)] = y + (offset * Dy)

    def chain(self, from_, to_):
        if from_ < 0 or to_ < 0 or from_ >= (
                self.img_x * self.img_y) or to_ >= (self.img_x * self.img_y):
            print("chain: one of the points is out of the image")
            return

        if from_ == to_:
            return 0.0

        if self.Ex[from_] < 0.0 or self.Ey[from_] < 0.0 or self.Ex[
                to_] < 0.0 or self.Ey[to_] < 0.0:
            return 0.0

        dx = self.Ex[to_] - self.Ex[from_]
        dy = self.Ey[to_] - self.Ey[from_]

        if (self.Gy[from_] * dx - self.Gx[from_] * dy) * (
                self.Gy[to_] * dx - self.Gx[to_] * dy) <= 0.0:
            return 0.0

        if (self.Gy[from_] * dx - self.Gx[from_] * dy) >= 0.0:
            return 1.0 / dist(self.Ex[from_], self.Ey[from_], self.Ex[to_],
                              self.Ey[to_])
        else:
            return -1.0 / dist(self.Ex[from_], self.Ey[from_], self.Ex[to_],
                               self.Ey[to_])

    def run(self):
        if self.sigma != 0.0:
            self.image_filtered =\
                self.image.filter(ImageFilter.GaussianBlur(radius=self.sigma))
            self.image_vec =\
                np.ravel(np.array(self.image_filtered, dtype=np.float64)).reshape([self.img_x * self.img_y, 1])

        self.compute_gradient()

        self.compute_edge_points()
        self.chain_edge_points()
        self.thresholds_with_hysteresis()
        self.list_chained_edge_points()

        if len(self.edges_x) != len(self.edges_y):
            print("detect edges: size of x and y edges mismatch")
            return

        return [self.edges_x, self.edges_y]
