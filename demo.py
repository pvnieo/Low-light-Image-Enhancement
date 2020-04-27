# std
import argparse
from argparse import RawTextHelpFormatter
import glob
from os import makedirs
from os.path import join, exists, basename, splitext
# 3p
import cv2
from tqdm import tqdm
# project
from exposure_enhancement import enhance_image_exposure


def main(args):
    # load images
    imdir = args.folder
    ext = ['png', 'jpg', 'bmp']    # Add image formats here
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    images = [cv2.imread(file) for file in files]

    # create save directory
    directory = join(imdir, "enhanced")
    if not exists(directory):
        makedirs(directory)

    # enhance images
    for i, image in tqdm(enumerate(images), desc="Enhancing images"):
        enhanced_image = enhance_image_exposure(image, args.gamma, args.lambda_, not args.lime,
                                                sigma=args.sigma, bc=args.bc, bs=args.bs, be=args.be, eps=args.eps)
        filename = basename(files[i])
        name, ext = splitext(filename)
        method = "LIME" if args.lime else "DUAL"
        corrected_name = f"{name}_{method}_g{args.gamma}_l{args.lambda_}{ext}"
        cv2.imwrite(join(directory, corrected_name), enhanced_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python implementation of two low-light image enhancement techniques via illumination map estimation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-f", '--folder', default='./demo/', type=str,
                        help="folder path to test images.")
    parser.add_argument("-g", '--gamma', default=0.6, type=float,
                        help="the gamma correction parameter.")
    parser.add_argument("-l", '--lambda_', default=0.15, type=float,
                        help="the weight for balancing the two terms in the illumination refinement optimization objective.")
    parser.add_argument("-ul", "--lime", action='store_true',
                        help="Use the LIME method. By default, the DUAL method is used.")
    parser.add_argument("-s", '--sigma', default=3, type=int,
                        help="Spatial standard deviation for spatial affinity based Gaussian weights.")
    parser.add_argument("-bc", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's contrast measure.")
    parser.add_argument("-bs", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's saturation measure.")
    parser.add_argument("-be", default=1, type=float,
                        help="parameter for controlling the influence of Mertens's well exposedness measure.")
    parser.add_argument("-eps", default=1e-3, type=float,
                        help="constant to avoid computation instability.")

    args = parser.parse_args()
    main(args)
