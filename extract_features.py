import dsift
import os, argparse
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize
from skimage import io, color, exposure
from scipy import misc
import matplotlib.pyplot as plt

def find_smallest_shape(dataset, extensionsToCheck):
    img_new_shape = [float("inf"), float("inf")]
    classes = next(os.walk(dataset))[1]

    for ii in classes:
        img_dir = '%s/%s' % (dataset, ii)
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
    parser.add_argument('-f', '--feature_extraction', required=False, default="HOG", help="name of the feature extraction method that will be used (Possible methods: 'HOG', 'LBP', ...).", choices=['HOG', 'LBP'])
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

    if(args.feature_extraction == "HOG"):
        # Deals with images with different sizes (descritors should have the same size)
        new_shape = find_smallest_shape(args.dataset, extensionsToCheck)

    for ii in classes:
        img_dir = '%s/%s' % (args.dataset, ii)
        img_files = ['%s/%s' % (img_dir, img) for img in os.listdir(img_dir) if any(ext in img for ext in extensionsToCheck)]

        for jj in img_files:
            img = io.imread(jj, 1)

            if(args.feature_extraction == "HOG"):
                img_gray = color.rgb2gray(img)

                if img_gray.shape[0] <= img_gray.shape[1]:
                    img_gray = resize(img_gray, new_shape)
                else:
                    img_gray = resize(img_gray, [new_shape[1], new_shape[0]])

                fd, hog_img = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),                              cells_per_block=(1, 1), visualise=True)

                fv_matrix = np.vstack([fv_matrix, fd]) if fv_matrix.size else fd

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

            labels.append(ii)
            img_names.append(jj.split('/')[-1])

    columns = ['Feat'+str(i) for i in range(1, fv_matrix.shape[1]+1)]
    df = pd.DataFrame(data=fv_matrix, index=img_names, columns=columns)

    df['Labels'] = labels

    # set dataset_name as PARENTFOLDER_IMAGESFOLDER
    dataset_name = "%s_%s" % (args.dataset.rstrip('/').split('/')[-2], args.dataset.rstrip('/').split('/')[-1])
    hdf = pd.HDFStore("%s/%s.%s" % (args.dataset.rstrip('/').rpartition('/')[0].rstrip('/'), dataset_name, 'h5'))

    hdf.put(args.feature_extraction, df, data_columns=True)
    hdf.close()


if __name__ == '__main__':
    # Sample usage
    #
    # python extract_features.py -d /path/to/images/folder/ -f LBP
    extract_features()
