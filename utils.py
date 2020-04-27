# 3p
import numpy as np


def get_derivation_op_fft():
    pass


# Convert point-spread function to optical transfer function


def psf2otf(psf, outSize=None):
    # Prepare psf for conversion
    data = prepare_psf(psf, outSize)

    # Compute the OTF
    otf = np.fft.fftn(data)

    return np.complex64(otf)


def prepare_psf(psf, outSize=None, dtype=None):
    if not dtype:
        dtype = np.float32

    psf = np.float32(psf)

    # Determine PSF / OTF shapes
    psfSize = np.int32(psf.shape)
    if not outSize:
        outSize = psfSize
    outSize = np.int32(outSize)

    # Pad the PSF to outSize
    new_psf = np.zeros(outSize, dtype=dtype)
    new_psf[: psfSize[0], : psfSize[1]] = psf[:, :]
    psf = new_psf

    # Circularly shift the OTF so that PSF center is at (0,0)
    shift = -(psfSize // 2)
    psf = circshift(psf, shift)

    return psf


# Circularly shift array


def circshift(A, shift):
    for i in range(shift.size):
        A = np.roll(A, shift[i], axis=i)
    return A


def get_sparse_neighbor(p: int, n: int, m: int):
    """Returns a dictionnary, where the keys are index of 4-neighbor of `p` in the sparse matrix,
       and values are tuples (i, j, x), where `i`, `j` are index of neighbor in the normal matrix,
       and x is the direction of neighbor.

    Arguments:
        p {int} -- index in the sparse matrix.
        n {int} -- number of rows in the original matrix (non sparse).
        m {int} -- number of columns in the original matrix.

    Returns:
        dict -- dictionnary containing indices of 4-neighbors of `p`.
    """
    i, j = p // m, p % m
    d = {}
    if i - 1 >= 0:
        d[(i - 1) * m + j] = (i - 1, j, 0)
    if i + 1 < n:
        d[(i + 1) * m + j] = (i + 1, j, 0)
    if j - 1 >= 0:
        d[i * m + j - 1] = (i, j - 1, 1)
    if j + 1 < m:
        d[i * m + j + 1] = (i, j + 1, 1)
    return d
