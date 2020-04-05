# stdlib

# 3p
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


class ExposureCorrector:
    def __init__(self, sigma=3, exp_sigma=0.2, bc=1, bs=1, be=1):  # default parameters suggested in the papers
        self.exposure_sigma = exp_sigma  # sigma used in computing the well_exposedness map
        self.bc, self.bs, self.be = bc, bs, be  # parameters for controlling the influence of Mertens's metrics
        self.eps = 1e-3  # for computational stability

        # create spatial affinity kernel
        self.kernel = np.zeros((15, 15))
        for i in range(15):
            for j in range(15):
                self.kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (7, 7)) ** 2) / (sigma ** 2))

    def get_smoothness_weights(self, L, x):
        # returns the smoothness weights according to the direction x
        Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=3)
        T = convolve(np.ones_like(L), self.kernel, mode='constant')
        T = T / (np.abs(convolve(Lp, self.kernel, mode='constant')) + self.eps)
        return T / (np.abs(Lp) + self.eps)

    def get_sparse_neighbor(self, p, n, m):
        # return dict, where
        # key p: is index in the sparse matrix (nm x nm)
        # value (i, j, x): (i, j) index in 2D matrix, x indicate if neighbor is in the x axis
        i, j = p // m, p % m
        d = {}
        if i-1 >= 0:
            d[(i-1) * m + j] = (i-1, j, 0)
        if i+1 < n:
            d[(i+1) * m + j] = (i+1, j, 0)
        if j-1 >= 0:
            d[i * m + j - 1] = (i, j - 1, 1)
        if j+1 < m:
            d[i * m + j + 1] = (i, j + 1, 1)
        return d

    # Mertens' measures
    def contrast_measure(self, im):
        depth = cv2.CV_16S
        kernel_size = 3
        im = (im * 255).astype('uint8')
        src = cv2.GaussianBlur(im, (3, 3), 0)
        src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(src_gray, depth, ksize=kernel_size)
        return np.abs(laplacian)

    def saturation_measure(self, im):
        return np.std(im, axis=-1)

    def expo_measure(self, im):
        well_exposedness = np.exp(- (im - 0.5)**2 / (2 * self.exposure_sigma ** 2))
        return np.prod(well_exposedness, axis=-1)

    def visual_quality_map(self, im):
        # returns the visual quality map used by DUAL to fuse the images (original, corrected_over, corrected_under)
        C = self.contrast_measure(im)
        S = self.saturation_measure(im)
        E = self.expo_measure(im)
        return (C ** self.bc) * (S ** self.bs) * (E ** self.be)

    def refine_illumination_map(self, L, gamma, lambda_):
        # perform the refinement of an initial illumination map as it's described in DUAL and LIME
        # gamma: is the gamma correction parameter
        # lambda_: is the weight for balancing the two terms in the optimization objective
        n, m = L.shape
        L_1d = L.copy().flatten()

        # compute smoothness weights
        wx = self.get_smoothness_weights(L, x=1)
        wy = self.get_smoothness_weights(L, x=0)

        # compute the five-point spatially inhomogeneous Laplacian matrix
        # maybe depending on way of defining the forward difference operator
        # if true, the expression may change
        # TODO: investigate this
        row, column, data = [], [], []
        for p in range(n*m):
            diag = 0
            for q, (k, l, x) in self.get_sparse_neighbor(p, n, m).items():
                weight = wx[k, l] if x else wy[k, l]
                row.append(p)
                column.append(q)
                data.append(-weight)
                diag += weight
            row.append(p)
            column.append(p)
            data.append(diag)

        F = csr_matrix((data, (row, column)), shape=(n*m, n*m))

        # solve the linear system
        Id = diags([np.ones(n*m)], [0])
        A = Id + lambda_ * F
        L_refined = spsolve(csr_matrix(A), L_1d).reshape((n, m))

        # gamma correction
        L_refined = L_refined ** gamma

        return L_refined

    def enhance_image(self, im, gamma, lambda_):
        # first estimation of the illumination map
        L = np.max(im, axis=-1)
        # illumination refinement
        L_refined = self.refine_illumination_map(L, gamma, lambda_)

        # correct image exposure
        L_refined_3d = np.repeat(L_refined[:, :, None], 3, axis=-1)
        im_enhanced = im / L_refined_3d
        return im_enhanced

    def fuse_images(self, im, im_enhanced, inv_im_enhanced):
        # compute visual quality map
        V0 = self.visual_quality_map(im)
        V1 = self.visual_quality_map(im_enhanced)
        V2 = self.visual_quality_map(inv_im_enhanced)

        # fusion
        V = np.repeat(np.argmax(np.stack([V0, V1, V2]), axis=0)[:, :, None], 3, axis=-1)
        return im * (V == 0) + im_enhanced * (V == 1) + inv_im_enhanced * (V == 2)

    def correct(self, im, gamma=0.8, lambda_=1, lime=False):
        # TODO: resize image if too large, optimization take too much time
        # TODO: add sigma to hyperparameters

        # correct exposure used the selected method (DUAL or LIME)
        # takes as an input a 8bit encoded image
        im_normalized = im.astype(float) / 255.
        inv_im_normalized = 1 - im_normalized

        # enchance images
        im_enhanced = self.enhance_image(im_normalized, gamma, lambda_)

        if not lime:
            inv_im_enhanced = 1 - self.enhance_image(inv_im_normalized, gamma, lambda_)
            # fuse images
            im_corrected = self.fuse_images(im_normalized, im_enhanced, inv_im_enhanced)
        else:
            im_corrected = im_enhanced

        # convert to 8 bits
        im_corrected = (np.clip(im_corrected, 0, 1) * 255).astype("uint8")

        return im_corrected
