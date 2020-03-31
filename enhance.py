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
from exposure_corrector import ExposureCorrector


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

    # create enhancer
    corrector = ExposureCorrector()

    # enhance images
    for i, image in tqdm(enumerate(images), desc="Enhancing images"):
        enhanced_image = corrector.correct(image, args.gamma, args.lambda_, args.lime)
        filename = basename(files[i])
        name, ext = splitext(filename)
        method = "LIME" if args.lime else "DUAL"
        corrected_name = f"{name}_{method}_g{args.gamma}_l{args.lambda_}.{ext}"
        cv2.imwrite(join(directory, corrected_name), enhanced_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Python implementation of two low-light image enhancement techniques via illumination map estimation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-f", '--folder', default='./demo/', type=str,
                        help="folder path to test images.")
    parser.add_argument("-g", '--gamma', default=0.8, type=float,
                        help="the gamma correction parameter.")
    parser.add_argument("-l", '--lambda_', default=1.0, type=float,
                        help="the weight for balancing the two terms in the illumination refinement optimization objective.")
    parser.add_argument("-ul", "--lime", action='store_true',
                        help="Use the LIME method. By default, the DUAL method is used.")

    args = parser.parse_args()
    main(args)
