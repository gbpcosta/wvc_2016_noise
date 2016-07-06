# import cv2
import os
import numpy as np
from skimage.feature import hog
from skimage import io, color, exposure
import matplotlib.pyplot as plt

def print_hog_images(img_gray, hog_image, filename="hog.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img_gray, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.savefig(filename)

def extract_features():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', required=True,
        help="path to the dataset folder")
    parser.add_argument(
        '-f', '--feature_extraction', required=False,
        default="HOG",
        help="name of the feature extraction method that will be used (Possible methods: 'HOG', ...).")
    parser.add_argument(
        '-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    # Dataset folder should contain a separate folder for each class that contains the images from that class
    # dataset = "/home/gbpcosta/datasets/CorelDB"

    # Descriptor to be extracted
    # HOG
    # descriptor = "HOG"

    classes = next(os.walk(args.dataset))[1]

    for ii in classes:
        img_dir = '%s/%s' % (dataset, ii)
        img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir)]

        for jj in img_files:
            img = io.imread(jj, 1)

            if(args.feature_extraction == "HOG"):
                img_gray = color.rgb2gray(img)

                fd, hog_image = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),                              cells_per_block=(1, 1), visualise=True)

            # TODO: extract features of one image

if __name__ == '__main__':
    extract_features()
