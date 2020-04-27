# 3p
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
# project
from utils import get_sparse_neighbor


def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15):
    """Create a kernel (`size` * `size` matrix) that will be used to compute the he spatial affinity based Gaussian weights.

    Arguments:
        spatial_sigma {float} -- Spatial standard deviation.

    Keyword Arguments:
        size {int} -- size of the kernel. (default: {15})

    Returns:
        np.ndarray - `size` * `size` kernel
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

    return kernel


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
    """Compute the smoothness weights used in refining the illumination map optimization problem.

    Arguments:
        L {np.ndarray} -- the initial illumination map to be refined.
        x {int} -- the direction of the weights. Can either be x=1 for horizontal or x=0 for vertical.
        kernel {np.ndarray} -- spatial affinity matrix

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability. (default: {1e-3})

    Returns:
        np.ndarray - smoothness weights according to direction x. same dimension as `L`.
    """
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)


def fuse_multi_exposure_images(im: np.ndarray, under_ex: np.ndarray, over_ex: np.ndarray,
                               bc: float = 1, bs: float = 1, be: float = 1):
    """perform the exposure fusion method used in the DUAL paper.

    Arguments:
        im {np.ndarray} -- input image to be enhanced.
        under_ex {np.ndarray} -- under-exposure corrected image. same dimension as `im`.
        over_ex {np.ndarray} -- over-exposure corrected image. same dimension as `im`.

    Keyword Arguments:
        bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
        bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
        be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})

    Returns:
        np.ndarray -- the fused image. same dimension as `im`.
    """
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
    fused_images = merge_mertens.process(images)
    return fused_images


def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """Refine the illumination map based on the optimization problem described in the two papers.
       This function use the sped-up solver presented in the LIME paper.

    Arguments:
        L {np.ndarray} -- the illumination map to be refined.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3}).

    Returns:
        np.ndarray -- refined illumination map. same shape as `L`.
    """
    # compute smoothness weights
    wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    L_1d = L.copy().flatten()

    # compute the five-point spatially inhomogeneous Laplacian matrix
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    # solve the linear system
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    # gamma correction
    L_refined = np.clip(L_refined, eps, 1) ** gamma

    return L_refined


def correct_underexposure(im: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """correct underexposudness using the retinex based algorithm presented in DUAL and LIME paper.

    Arguments:
        im {np.ndarray} -- input image to be corrected.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3})

    Returns:
        np.ndarray -- image underexposudness corrected. same shape as `im`.
    """

    # first estimation of the illumination map
    L = np.max(im, axis=-1)
    # illumination refinement
    L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

    # correct image underexposure
    L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
    im_corrected = im / L_refined_3d
    return im_corrected

# TODO: resize image if too large, optimization take too much time


def enhance_image_exposure(im: np.ndarray, gamma: float, lambda_: float, dual: bool = True, sigma: int = 3,
                           bc: float = 1, bs: float = 1, be: float = 1, eps: float = 1e-3):
    """Enhance input image, using either DUAL method, or LIME method. For more info, please see original papers.

    Arguments:
        im {np.ndarray} -- input image to be corrected.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem (in DUAL and LIME).

    Keyword Arguments:
        dual {bool} -- boolean variable to indicate enhancement method to be used (either DUAL or LIME) (default: {True})
        sigma {int} -- Spatial standard deviation for spatial affinity based Gaussian weights. (default: {3})
        bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
        bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
        be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})
        eps {float} -- small constant to avoid computation instability (default: {1e-3})

    Returns:
        np.ndarray -- image exposure enhanced. same shape as `im`.
    """
    # create spacial affinity kernel
    kernel = create_spacial_affinity_kernel(sigma)

    # correct underexposudness
    im_normalized = im.astype(float) / 255.
    under_corrected = correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)

    if dual:
        # correct overexposure and merge if DUAL method is selected
        inv_im_normalized = 1 - im_normalized
        over_corrected = 1 - correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)
        # fuse images
        im_corrected = fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
    else:
        im_corrected = under_corrected

    # convert to 8 bits and returns
    return np.clip(im_corrected * 255, 0, 255).astype("uint8")
