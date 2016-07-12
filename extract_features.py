ls
# import cv2
import os
import numpy as np
from skimage.feature import hog, local_binary_pattern, daisy
from skimage import io, color, exposure
import matplotlib.pyplot as plt

def print_hog_images(img_gray, hog_img, filename="hog.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img_gray, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.savefig(filename)

def overlay_labels(img, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=img, bg_label=0, alpha=0.5)

def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

def print_lbp_image(img, lbp, filename="lbp.png"):
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()

    titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)

    for ax, labels in zip(ax_img, label_sets):
        ax.imshow(overlay_labels(img, lbp, labels))

    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
        highlight_bars(bars, labels)
        ax.set_ylim(ymax=np.max(counts[:-1]))
        ax.set_xlim(xmax=n_points + 2)
        ax.set_title(name)

    ax_hist[0].set_ylabel('Percentage')
    for ax in ax_img:
        ax.axis('off')

    plt.savefig(filename)

def print_daisy_image(daisy_img, daisy_descs, filename="daisy.png"):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(daisy_img)
    descs_num = daisy_descs.shape[0] * daisy_descs.shape[1]
    ax.set_title('%i DAISY descriptors extracted:' % descs_num)
    plt.savefig(filename)

def extract_features():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,        help="path to the dataset folder")
    parser.add_argument('-f', '--feature_extraction', required=False, default="HOG", help="name of the feature extraction method that will be used (Possible methods: 'HOG', 'LBP', 'DAISY', 'SIFT', ...).", choices=['HOG', 'LBP', 'DAISY', 'SIFT'])
    parser.add_argument('-s', '--save_image', help="Print image  comparison of the original image and a visualization of the descriptors extracted.", default=False)
    parser.add_argument('-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    classes = next(os.walk(args.dataset))[1]
    fv_matrix =

    for ii in classes:
        img_dir = '%s/%s' % (dataset, ii)
        img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir)]

        for jj in img_files:
            img = io.imread(jj, 1)

            if(args.feature_extraction == "HOG"):
                # TODO: deal with images with different sizes (descritors should have the same size)
                img_gray = color.rgb2gray(img)

                fd, hod_img = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),                              cells_per_block=(1, 1), visualise=True)

                fv_matrix = np.stack((fv_matrix, fd), axis=-1)

                if(args.save_image == True):
                    print_hog_images(img_gray, hog_img, filename="hog.png")

            elif(args.feature_extraction == "LBP"):
                img_gray = color.rgb2gray(img)

                # LBP Parameters
                radius = 3
                n_points = 8 * radius
                METHOD = 'uniform'

                lbp = local_binary_pattern(img_gray, n_points, radius, METHOD)

                if(args.save_image == True):
                    print_lbp_image(img_gray, lbp, filename="lbp.png")

            elif(args.feature_extraction == "DAISY"):
                img_gray = color.rgb2gray(img)

                descs, descs_img = daisy(img_gray, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)

                if(args.save_image == True):
                    print_lbp_image(img_gray, descs, filename="daisy.png")

            elif(args.feature_extraction == "SIFT")


            # TODO: extract features of one image
            # TODO: create matrix of descriptors (one image in line)

if __name__ == '__main__':
    extract_features()
