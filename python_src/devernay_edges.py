import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import time 


def dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))


def greater(a,b):
    if a <= b:
        return False
    
    if (a - b) < 1e-10:
        return False

    return True


class DevernayEdge:

    def __init__(self, image, sigma=0.0, high_thresh=0.0, low_thresh=0.0):
        self.sigma = sigma
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.image = image 
        #image details
        [self.img_y, self.img_x] = ImageOps.grayscale(self.image).size
        print("(grayscale) Image size:", [self.img_x,self.img_y])
        self.image = np.ravel(np.array(self.image, dtype=np.float64)).reshape([self.img_x*self.img_y, 1])
        
        self.Gx   = np.empty([self.img_x*self.img_y, 1])
        self.Gy   = np.empty([self.img_x*self.img_y, 1])
        self.modG = np.empty([self.img_x*self.img_y, 1])

        self.Ex = -np.ones((self.img_x*self.img_y, 1), dtype=np.float64)
        self.Ey = -np.ones((self.img_x*self.img_y, 1), dtype=np.float64)

        self.next_ = -np.ones((self.img_x*self.img_y, 1), dtype=np.float64)
        self.prev_ = -np.ones((self.img_x*self.img_y, 1), dtype=np.float64)

        self.edges_x = []
        self.edges_y = []


    
    def compute_gradient(self):
        for x in range(1, self.img_x-1):
            for y in range(1, self.img_y-1):
                self.Gx[x+y*self.img_x]   = np.float64(self.image[(x+1)+(y*self.img_x)] - self.image[(x-1)+(y*self.img_x)])
                self.Gy[x+y*self.img_x]   = np.float64(self.image[x+((y+1)*self.img_x)] - self.image[x+((y-1)*self.img_x)])
                self.modG[x+y*self.img_x] = np.float64(np.sqrt(self.Gx[x+(y*self.img_x)] * self.Gx[x+(y*self.img_x)] + self.Gy[x+(y*self.img_x)] * self.Gy[x+(y*self.img_x)]))



    def list_chained_edge_points(self):
        N = 0 
        M = 0
        curve_limits = []

        for i in range(0, self.img_x * self.img_y):
            if self.prev_[i] >= 0 or self.next_[i] >= 0:
                # curve_limits[M] = np.int32(N) 
                curve_limits.insert(M, np.int32(N))
                M += 1

                k = i 
                n = self.prev_[k]

                while n >= 0 and n != i:
                    k = n 
                    n = self.prev_[k]

                while True:
                    # self.edges_x[N] = self.Ex[k]
                    # self.edges_y[N] = self.Ey[k]
                    self.edges_x.insert(N, self.Ex[k])
                    self.edges_y.insert(N, self.Ey[k])

                    N += 1
                    n = self.next_[k]
                    self.next_[k] = -1 
                    self.prev_[k] = -1 
                    k = n 
                    if k < 0:
                        break
        
        # curve_limits[M] = np.int32(N) 
        curve_limits.insert(M, np.int32(N))


                    
    def chain_edge_points(self):
        if len(self.next_) == 0 or len(self.prev_) == 0 or len(self.Ex) == 0 or len(self.Ey) == 0 or len(self.Gx)== 0 or len(self.Gy) == 0:
            print("chain_edge_points: invalid input")
            return 
        
        for x in range(2, self.img_x-2):
            for y in range(2, self.img_x-2):
                if self.Ex[x+y*(self.img_x-1)] >= 0.0 and self.Ey[x+y*(self.img_x-1)] >= 0.0:
                    from_ = np.int32(x + y * self.img_x)
                    fwd_s = 0.0
                    bck_s = 0.0
                    fwd = -1 
                    bck = -1

                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            to_ = np.int32(x + i + (y+j) * self.img_x)
                            s = self.chain(np.int32(from_), np.int32(to_))

                            if s > fwd_s:
                                fwd_s = s 
                                fwd = to_
                            
                            if s < bck_s:
                                bck_s = s 
                                bck = to_ 
                            
                    if fwd >= 0 and self.next_[from_] != fwd:
                        alt = np.int32(self.prev_[fwd])
                        if alt < 0 or self.chain(np.int32(alt), np.int32(fwd)) < fwd_s:
                            if self.next_[from_] >= 0:
                                self.prev_[np.int32(self.next_[from_])] = -1 
                            self.next_[from_] = fwd

                            if alt >= 0:
                                self.next_[alt] = -1 
                            self.prev_[fwd] = from_

                    if bck >= 0 and self.prev_[from_] != bck:
                        alt = np.int32(self.next_[bck])
                        if alt < 0 or self.chain(np.int32(alt), np.int32(bck)) > bck_s:
                            if alt >= 0:
                                self.prev_[alt] = -1 
                            self.next_[bck] = -1 
                        
                            if self.prev_[from_] >= 0:
                                self.next_[np.int32(self.prev_[from_])] = -1 
                            self.prev_[from_] = bck 



    def thresholds_with_hysteresis(self):
        if len(self.modG) == 0 or len(self.next_) == 0 or len(self.prev_) == 0:
            print("threhold_with_hysteresis: invalid input")
            return

        valid = np.full((self.img_x*self.img_y, 1), False)

        for i in range(0, self.img_x*self.img_y):
            if (self.prev_[i] >= 0 or self.next_[i] >= 0) and not valid[i] and self.modG[i] >= self.high_thresh:
                valid[i] = True

                j = i
                k = np.int32(self.next_[j])
                while j >= 0 and k >= 0 and (not valid[k]):
                    if self.modG[k] < self.low_thresh:
                        self.next_[j] = -1 
                        self.prev_[k] = -1
                    else:
                        valid[k] = True

                    #TODO: check casting here again later..should I use squeeze??
                    j = np.squeeze(np.int32(self.next_[j]))
                    k = np.squeeze(np.int32(self.next_[j]))


                j = i 
                k = np.int32(self.prev_[j])
                while j >= 0 and k >= 0 and (not valid[k]):
                    if self.modG[k] < self.low_thresh:
                        self.prev_[j] = -1 
                        self.prev_[k] = -1 
                    else:
                        valid[k] = True

                    j = np.squeeze(np.int32(self.prev_[j]))
                    k = np.squeeze(np.int32(self.prev_[j]))


        for i in range(0, self.img_x * self.img_y):
            if self.prev_[i] >= 0 or self.next_[i] >= 0 and (not valid[i]):
                self.prev_[i] = -1 
                self.next_[i] = -1 



    def compute_edge_points(self):
        if len(self.Ex) == 0 or len(self.Ey) == 0 or len(self.modG) == 0 or len(self.Gx) == 0 or len(self.Gy) == 0:
            print("compute_edge_points: invalid input")
            return 

        for x in range(2, self.img_x-2):
            for y in range(2, self.img_y-2):
                Dx = 0
                Dy = 0

                mod = self.modG[x+y*self.img_x]
                L =   self.modG[x-1+y*self.img_x]
                R =   self.modG[x+1+y*self.img_x]
                U =   self.modG[x+(y+1)*self.img_x]
                D =   self.modG[x+(y-1)*self.img_x]
                gx =  np.fabs(self.Gx[x+y*self.img_x])
                gy =  np.fabs(self.Gy[x+y*self.img_x])

                if greater(mod, L) and not greater(R, mod) and gx >= gy:
                    Dx = 1
                elif greater(mod,D) and not greater(U, mod) and gx <= gy:
                    Dy = 1
                
                if Dx > 0 or Dy > 0:
                    a = self.modG[x - Dx + (y-Dy) * self.img_x]
                    b = self.modG[x + y * self.img_x]
                    c = self.modG[x + Dx + (y+Dy) * self.img_x]
                    offset = 0.5 * (a-c) / (a-b-b+c)

                    self.Ex[x+y*self.img_x] = x + offset * Dx
                    self.Ey[x+y*self.img_x] = y + offset * Dy



    def chain(self, from_, to_):

        if len(self.Gx) == 0 or len(self.Gy) == 0 or len(self.Ex) == 0 or len(self.Ey) == 0:
            print("chain: invalid input")
            return 
        
        #TODO: change end idx of image vec..
        if from_ < 0 or to_ < 0 or from_ > (self.img_x * self.img_y - 1) or to_ > (self.img_x * self.img_y - 1):
            print("chain: one of the points is out of the image")
            return 

        if from_ == to_:
            return 0.0
        

        if self.Ex[from_] < 0.0 or self.Ey[from_] < 0.0 or self.Ex[to_] < 0.0 or self.Ey[to_] < 0.0:
            return 0.0
        
        dx = np.float64(self.Ex[to_] - self.Ex[from_])
        dy = np.float64(self.Ey[to_] - self.Ey[from_])

        if (self.Gy[from_] * dx - self.Gx[from_] * dy) * (self.Gy[to_] * dx - self.Gx[to_] * dy) <= 0.0:
            return 0.0
        
        if (self.Gy[from_] * dx - self.Gx[from_] * dy) >= 0.0:
            return  np.float64(1.0 / dist(self.Ex[from_], self.Ey[from_], self.Ex[to_], self.Ey[from_]))
        else:
            return np.float64(-1.0 / dist(self.Ex[from_], self.Ey[from_], self.Ex[to_], self.Ey[from_]))
        


    def detect_edges(self):
        if self.sigma == 0:
            self.compute_gradient()
            print("compute_gradient without gauss filter finished..")
        else:
            # self.image = self.image.filter(ImageFilter.GaussianBlur(radius = self.sigma))
            self.image = gaussian_filter(self.image, sigma=self.sigma)
            self.compute_gradient()
            print("compute_gradient with gauss filter finished..")
        

        self.compute_edge_points()
        print("compute_edge_points finished..")
        
        self.chain_edge_points()
        print("chain_edge_points finished..")

        self.thresholds_with_hysteresis()
        print("thresholds_with_hysteresis finished..")

        self.list_chained_edge_points()
        print("list_chained_edge_points..")

        if len(self.edges_x) != len(self.edges_y):
            print("x and y edges size mismatch")
            return 
        return [self.edges_x, self.edges_y]


def main():
    
    sigma = 0.0
    high_treshold = 0.0
    low_threshold = 0.0

    image_rgb = Image.open(r"cat.jpg")    
    image_binary = Image.open(r"binary_cat.jpg")

    start = time.time()
    devernayEdge = DevernayEdge(image_binary, sigma, high_treshold, low_threshold)
    [edges_x, edges_y] = devernayEdge.detect_edges()
    stop = time.time()
    print("Elapsed time:", stop-start, " secs")

    print("edges_x len:", len(edges_x))
    print("edges_y len:", len(edges_y))

    plt.figure(1)
    plt.title("Devernay Edge Detection")
    plt.imshow(np.array(image_rgb), cmap="gray")
    plt.scatter(edges_x, edges_y, color="magenta", marker=".", linewidth=.1)

    plt.figure(2)
    plt.imshow(image_binary)

    plt.show()


if __name__ == "__main__": main()



# image_gray = ImageOps.grayscale(image)
# print("image size:" , image_gray.size)

# #image_binary = image_gray.convert("1")
# image_binary = np.array(image_gray)
# image_binary[image_binary > 250] = 255
# image_binary[image_binary < 250] = 0

# # plt.imshow(image_binary)
# # plt.show()