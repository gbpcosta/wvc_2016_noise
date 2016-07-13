# import cv2
import dsift
import os, argparse
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern, daisy
from skimage.transform import resize
from skimage import io, color, exposure
from scipy import misc
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

def find_smallest_shape(dataset, extensionsToCheck):
    img_new_shape = [float("inf"), float("inf")]
    classes = next(os.walk(dataset))[1]

    for ii in classes:
        img_dir = '%s/%s' % (dataset, ii)
        # img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir)]
        img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir) if any(ext in img for ext in extensionsToCheck)]

        for jj in img_files:
            img = io.imread(jj, 1)

            shape_min = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
            shape_max = img.shape[0] if img.shape[0] > img.shape[1] else img.shape[1]

            img_new_shape[0] = shape_min if img_new_shape[0] > shape_min else img_new_shape[0]
            img_new_shape[1] = shape_max if img_new_shape[1] > shape_min else img_new_shape[1]

    return img_new_shape

def extract_features():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,        help="path to the dataset folder")
    parser.add_argument('-f', '--feature_extraction', required=False, default="HOG", help="name of the feature extraction method that will be used (Possible methods: 'HOG', 'LBP', 'SIFT', ...).", choices=['HOG', 'LBP',  'SIFT'])
    # TODO: set filename used to save images as parameter
    parser.add_argument('-s', '--save_image', help="Save image  comparison of the original image and a visualization of the descriptors extracted.", default=False)
    # TODO: use verbose parameter to track progress
    parser.add_argument('-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    extensionsToCheck = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

    classes = next(os.walk(args.dataset))[1]
    fv_matrix = np.array([])
    labels = []
    img_names = []

    if(args.feature_extraction == "HOG" or args.feature_extraction == "SIFT"):
        new_shape = find_smallest_shape(args.dataset, extensionsToCheck)

    for ii in classes:
        img_dir = '%s/%s' % (args.dataset, ii)
        # img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir)]
        img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir) if any(ext in img for ext in extensionsToCheck)]

        for jj in img_files:
            img = io.imread(jj, 1)

            if(args.feature_extraction == "HOG"):
                # TODO: deal with images with different sizes (descritors should have the same size)
                img_gray = color.rgb2gray(img)

                if img_gray.shape[0] <= img_gray.shape[1]:
                    img_gray = resize(img_gray, new_shape)
                else:
                    img_gray = resize(img_gray, [new_shape[1], new_shape[0]])

                fd, hod_img = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),                              cells_per_block=(1, 1), visualise=True)

                fv_matrix = np.vstack([fv_matrix, fd]) if fv_matrix.size else fd

                if(args.save_image == True):
                    print_hog_images(img_gray, hog_img, filename="hog.png")

            elif(args.feature_extraction == "LBP"):
                img_gray = color.rgb2gray(img)
                # LBP Parameters
                radius = 1 # 3
                n_points = 8 * radius
                METHOD = 'uniform'
                bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                lbp = local_binary_pattern(img_gray, n_points, radius, METHOD)
                lbp_hist = np.histogram(lbp, bins=bins, density=True)[0]

                fv_matrix = np.vstack([fv_matrix, lbp_hist]) if fv_matrix.size else lbp_hist

                if(args.save_image == True):
                    print_lbp_image(img_gray, lbp, filename="lbp.png")

            elif(args.feature_extraction == "DAISY"):
                # TODO: Fix bugs! Problems with the selected parameter values
                img_gray = color.rgb2gray(img)

                descs, descs_img = daisy(img_gray, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)

                fv_matrix = np.vstack([fv_matrix, descs]) if fv_matrix.size else descs

                if(args.save_image == True):
                    print_lbp_image(img_gray, descs, filename="daisy.png")

            # elif(args.feature_extraction == "ORB"):
            elif(args.feature_extraction == "SIFT"):
                img_gray = color.rgb2gray(img)
                img_gray = np.array(img_gray)

                if img_gray.shape[0] <= img_gray.shape[1]:
                    img_gray = resize(img_gray, new_shape)
                else:
                    img_gray = resize(img_gray, [new_shape[1], new_shape[0]])

                # Sample Usage:
                #     extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
                #     feaArr,positions = extractor.process_image(Image)
                # Source:  https://github.com/Yangqing/dsift-python

                extractor = dsift.DsiftExtractor(8,16,1)
                feaArr,positions = extractor.process_image(img_gray)

                feaArr = np.reshape(feaArr, feaArr.shape[0] * feaArr.shape[1])

                fv_matrix = np.vstack([fv_matrix, feaArr]) if fv_matrix.size else feaArr

            labels.append(ii)
            img_names.append(jj.split('/')[-1])


    columns = ['Feat'+str(i) for i in range(1, fv_matrix.shape[1]+1)]
    # index = ['Sample'+str(i) for i in range(1, fv_matrix.shape[0]+1)]
    df = pd.DataFrame(data=fv_matrix, index=img_names, columns=columns)
    df['Labels'] = labels

    dataset_name = args.dataset.split('/')[-2]
    hdf = pd.HDFStore("%s.%s" % (dataset_name, 'h5'))

    if(args.feature_extraction == "SIFT"):
        hdf.put(args.feature_extraction, df, data_columns=True)
    else:
        hdf.put(args.feature_extraction, df, format='table', data_columns=True)
    hdf.close()


if __name__ == '__main__':
    extract_features()
