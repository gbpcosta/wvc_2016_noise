import os, argparse
import numpy as np
import pandas as pd
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def crossclassify():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,        help="path to the dataset folder")
    parser.add_argument('-f', '--feature_extraction', required=True, help="Name of the feature extraction method that will be used (Possible methods: 'HOG', 'LBP', 'SIFT', ...).", choices=['HOG', 'LBP',  'SIFT'])
    parser.add_argument('-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    score = 'f1'

    hdf = pd.read_hdf(args.dataset, args.feature_extraction)

    X = hdf.drop("Labels", axis=1)
    y = hdf["Labels"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    if args.feature_extraction == "SIFT":
        pca = PCA(n_components=0.9)
        X_train = pca.fit_transform(X_train)
        if args.verbose >= 1 :
            print "PCA for dimensionality reduction"
            print "Number of components: %d" % pca.n_components_
            print "Total variance: %lf" % sum(pca.explained_variance_ratio_)

    hdf = pd.read_hdf(args.dataset, "%s_gridsearch" % args.feature_extraction)
    best_params = hdf['parameters'].get(hdf['f1_weighted'].idxmax())

    clf = svm.SVC()
    clf.set_params(**best_params)
    clf.fit(X_train, y_train)

    h5_dir = args.dataset.rpartition('/')[0]
    h5_files = ['%s/%s' % (h5_dir, h5) for h5 in os.listdir(h5_dir) if any(ext in h5 for ext in ['.h5'])]

    noise = []
    f1s = []

    for h5_file in h5_files:

        hdf = pd.read_hdf(h5_file, args.feature_extraction)

        X = hdf.drop("Labels", axis=1)
        y = hdf["Labels"]

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

        if args.feature_extraction == "SIFT":
            X_test = pca.transform(X_test)

        if args.verbose >= 1:
            print "Detailed classification report:"
            print
            print "The model is trained on the full development set."
            print "The scores are computed on the full evaluation set."
            print
        y_true, y_pred = y_test, clf.predict(X_test)

        f1 = f1_score(y_true, y_pred, average='weighted')
        noise.append(h5_file.rpartition('/')[2].partition('_')[2].rpartition('.')[0])
        f1s.append(f1)

        if args.verbose >= 1:
            print classification_report(y_true, y_pred, digits=6)
            print

        target_names = list(set(y_true))

        confmat = confusion_matrix(y_true, y_pred, labels=target_names)
        if args.verbose >= 2:
            print "Confusion matrix:"
            print confmat

        np.set_printoptions(precision=6)

        # Normalize the confusion matrix by row (i.e by the number of samples in each class)
        confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
        if args.verbose >= 2:
            print 'Normalized confusion matrix:'
            print cm_normalized

    df = pd.DataFrame(data=f1s, index=noise, columns=["F1-score"])
    hdf = pd.HDFStore(args.dataset)
    hdf.put("%s_crossclass" % args.feature_extraction, df, data_columns=True)
    hdf.close()

if __name__ == '__main__':
    crossclassify()
