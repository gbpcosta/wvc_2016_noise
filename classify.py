import argparse
import numpy as np
import pandas as pd
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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


def classify():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,        help="path to the dataset folder")
    parser.add_argument('-f', '--feature_extraction', required=True, help="Name of the feature extraction method that will be used (Possible methods: 'HOG', 'LBP', 'SIFT', ...).", choices=['HOG', 'LBP',  'SIFT'])
    parser.add_argument('-pcm', '--plot_confmat', default=False, help="Plot a visualization of the confusion matrix.")
    parser.add_argument('-nj', '--n_jobs', required=False, type=int, default=1, help="Number of jobs used during grid search.")
    # TODO: use verbose parameter to track progress
    parser.add_argument('-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}] # {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
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
            # print "Variance of each component: " + pca.explained_variance_ratio_

        X_test = pca.transform(X_test)

    if args.verbose >= 1:
        print "# Tuning hyper-parameters for %s" % score
        print

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='%s_weighted' % score, n_jobs=args.n_jobs, verbose=args.verbose)
    clf.fit(X_train, y_train)

    if args.verbose >= 1:
        print "Best parameters set found on development set:"
        print
        print clf.best_params_
        print
        print "Grid scores on development set:"
        print
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
        print

    params = [param for param, mean_score, scores in clf.grid_scores_]
    mean_scores = [mean_score for param, mean_score, scores in clf.grid_scores_]
    scores_std = [scores.std() for param, mean_score, scores in clf.grid_scores_]

    df = pd.DataFrame({'f1_weighted': mean_scores, 'std': scores_std, 'parameters': params})
    dataset_name = args.dataset.rpartition('/')[2][:-3]
    hdf = pd.HDFStore("%s/%s.%s" % (args.dataset.rpartition('/')[0], dataset_name, 'h5'))
    hdf.put("%s_gridsearch" % args.feature_extraction, df, data_columns=True)
    hdf.close()

    if args.verbose >= 1:
        print "Detailed classification report:"
        print
        print "The model is trained on the full development set."
        print "The scores are computed on the full evaluation set."
        print
    y_true, y_pred = y_test, clf.predict(X_test)
    if args.verbose >= 1:
        print classification_report(y_true, y_pred)
        print

    predmat = np.vstack((y_true, y_pred))
    cols = ["%s_%s" % (y_true[i], X_test.index.values[i]) for i in xrange(len(y_true))]
    df = pd.DataFrame(data=predmat, index=["True", "Prediction"], columns=cols)
    dataset_name = args.dataset.rpartition('/')[2][:-3]
    hdf = pd.HDFStore("%s/%s.%s" % (args.dataset.rpartition('/')[0], dataset_name, 'h5'))
    hdf.put("%s_prediction" % args.feature_extraction, df, data_columns=True)
    hdf.close()

    target_names = list(set(y_true))
    # print_names = [abrev_names[x] for x in target_names]

    confmat = confusion_matrix(y_true, y_pred, labels=target_names)
    if args.verbose >= 2:
        print "Confusion matrix:"
        print confmat

    df = pd.DataFrame(data=confmat, index=target_names, columns=target_names)
    dataset_name = args.dataset.rpartition('/')[2][:-3]
    hdf = pd.HDFStore("%s/%s.%s" % (args.dataset.rpartition('/')[0], dataset_name, 'h5'))
    hdf.put("%s_confmat" % args.feature_extraction, df, data_columns=True)
    hdf.close()

    np.set_printoptions(precision=4)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
    if args.verbose >= 2:
        print 'Normalized confusion matrix:'
        print cm_normalized

    if args.plot_confmat == True:
        filename = "%s/confmat_%s_%s.png" % (args.dataset.rpartition('/')[0], dataset_name, args.feature_extraction)
        plot_confusion_matrix(confmat, target_names, filename=filename, title='Normalized confusion matrix')


if __name__ == '__main__':
    classify()
