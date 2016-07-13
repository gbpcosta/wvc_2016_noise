import argparse
import numpy as np
import pandas as pd
from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


def classify():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,        help="path to the dataset folder")
    parser.add_argument('-f', '--feature_extraction', required=True, help="name of the feature extraction method that will be used (Possible methods: 'HOG', 'LBP', 'SIFT', ...).", choices=['HOG', 'LBP',  'SIFT'])
    parser.add_argument('-nj', '--n_jobs', required=False, default=1, help="Number of jobs used during grid search.")
    # TODO: use verbose parameter to track progress
    parser.add_argument('-v', '--verbose', action='count', help="verbosity level.")
    args = parser.parse_args()

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    score = 'f1'

    hdf = pd.read_hdf("%s.h5" % args.dataset, args.feature_extraction)

    X = np.array(hdf.drop("Labels", axis=1), np.float)
    y = np.array(hdf["Labels"], np.str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print "# Tuning hyper-parameters for %s" % score
    print

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5, scoring='%s_weighted' % score, n_jobs=args.n_jobs, verbose=args.verbose)
    clf.fit(X_train, y_train)

    print "Best parameters set found on development set:"
    print
    print clf.best_params_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
    print

    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)
    print



if __name__ == '__main__':
    classify()
