import os, argparse
import numpy as np
import pandas as pd
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_confusion_matrix(confmat, confmat_norm, target_names, filepath, title='Confusion matrix', cmap=plt.cm.Blues):
    confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

    plt.figure(num=None, figsize=(15, 13), dpi=300, facecolor='w') #, edgecolor='k')

    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize='xx-large')
    plt.clim(vmin=0.0, vmax=1.0)
    width, height = cm_norm.shape

    for x in xrange(width):
        for y in xrange(height):
            if cm_norm[x][y] > 0.7:
                plt.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white', fontsize=24)
            else:
                plt.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='black', fontsize=24)

    cbar = plt.colorbar(aspect=20, fraction=.12,pad=.02)
    cbar.ax.tick_params(labelsize=20)

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=60, fontsize=24)
    plt.yticks(tick_marks, target_names, fontsize=24)

    plt.tight_layout()

    plt.ylabel('True label', fontsize=32)
    plt.xlabel('Predicted label', fontsize=32)

    plt.savefig(filepath, bbox_inches='tight')


def crossclassify():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,        help="path to the dataset folder")
    parser.add_argument('-f', '--feature_extraction', required=True, help="Name of the feature extraction method that will be used (Possible methods: 'HOG', 'LBP', 'SIFT', ...).", choices=['HOG', 'LBP',  'SIFT'])
    parser.add_argument('-pcm', '--plot_confmat', default=False, help="Plot a visualization of the confusion matrix.")
    # TODO: use verbose parameter to track progress
    parser.add_argument('-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    # tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}] # {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
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

    # TODO: find best params
    hdf = pd.read_hdf(args.dataset, "%s_gridsearch" % args.feature_extraction)
    best_params = hdf['parameters'].get(hdf['f1_weighted'].idxmax())


    # TODO: train svm with X_train and best_params
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
        # print_names = [abrev_names[x] for x in target_names]

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

        if args.plot_confmat == True:
            filename = "%s/confmat_%s_%s.png" % (args.dataset.rpartition('/')[0], dataset_name, args.feature_extraction)
            plot_confusion_matrix(confmat, target_names, filename=filename, title='Normalized confusion matrix')

    df = pd.DataFrame(data=f1s, index=noise, columns=["F1-score"])
    hdf = pd.HDFStore(args.dataset)
    hdf.put("%s_crossclass" % args.feature_extraction, df, data_columns=True)
    hdf.close()

if __name__ == '__main__':
    crossclassify()
