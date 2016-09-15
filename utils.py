import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_confusion_matrix(confmat, target_names, filepath, title='Confusion matrix', cmap=plt.cm.Blues):
    confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]

    plt.figure(num=None, figsize=(15, 13), dpi=300, facecolor='w') #, edgecolor='k')

    plt.imshow(confmat_norm, interpolation='nearest', cmap=cmap)
    # plt.title(title, fontsize='xx-large')
    plt.clim(vmin=0.0, vmax=1.0)
    width, height = confmat_norm.shape

    for x in xrange(width):
        for y in xrange(height):
            if cm_norm[x][y] > 0.7:
                plt.annotate(str(confmat[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white', fontsize=24)
            else:
                plt.annotate(str(confmat[x][y]), xy=(y, x),
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

def create_crossclassify_table(h5_dir,
    feature_extraction,
    select=None, verbose=0):

    h5_files = ['%s/%s' % (h5_dir, h5) for h5 in os.listdir(h5_dir) if any(ext in h5 for ext in ['.h5'])]

    ii = 0
    cols = []

    for h5_file in h5_files:
        if verbose > 0:
            print "Processing file %s..." % h5_file
        cols.append(h5_file.rpartition('/')[2].partition('_')[2].rpartition('.')[0])

        if ii == 0:
            hdf = pd.read_hdf(h5_file, '%s_crossclass' % feature_extraction)
            ii = 1
        else:
            hdf2 = pd.read_hdf(h5_file, '%s_crossclass' % feature_extraction)
            hdf = pd.concat([hdf, hdf2], axis=1)

    hdf.columns = cols

    if select is not None:
        hdf = hdf[select]
        hdf = hdf.T
        hdf = hdf[select]
    else:
        hdf = hdf.T

    return hdf

def save_crossclassify_heatmap(h5_dir,
    feature_extraction_methods,
    filename = '%s_%s.png',
    cmap = plt.cm.Greens,
    select= ['original',
     'gaussian-10',
     'gaussian-20',
     'gaussian-30',
     'gaussian-40',
     'gaussian-50',
     'NLM-25-gaussian-10',
     'NLM-25-gaussian-20',
     'NLM-25-gaussian-30',
     'NLM-25-gaussian-40',
     'NLM-25-gaussian-50',
     'poisson-10',
     'poisson-10.5',
     'poisson-11',
     'poisson-11.5',
     'poisson-12',
     'NLM-25-poisson-10',
     'NLM-25-poisson-10.5',
     'NLM-25-poisson-11',
     'NLM-25-poisson-11.5',
     'NLM-25-poisson-12',
     'sp-0.1',
     'sp-0.2',
     'sp-0.3',
     'sp-0.4',
     'sp-0.5',
     'Median-11-sp-0.1',
     'Median-11-sp-0.2',
     'Median-11-sp-0.3',
     'Median-11-sp-0.4',
     'Median-11-sp-0.5'],
    vmin=None,
    vmax=None):

    dataset_name = filter(None, h5_dir.split('/'))[-1]

    hdfs = {}

    if not isinstance(feature_extraction_methods, list):
        feature_extraction_methods = [feature_extraction_methods]

    for feature_extraction in feature_extraction_methods:
        if select is None:
            hdfs[feature_extraction] = create_crossclassify_table(h5_dir, feature_extraction)
        else:
            hdfs[feature_extraction] = create_crossclassify_table(h5_dir, feature_extraction, select)

    for ii, feature_extraction in enumerate(feature_extraction_methods):

        plt.figure(num=None, figsize=(15, 13), dpi=300, facecolor='w') #, edgecolor='k')
        if isinstance(cmap, list):
            plt.imshow(hdfs[feature_extraction], interpolation='nearest', cmap=cmap[ii])
        else:
          plt.imshow(hdfs[feature_extraction], interpolation='nearest', cmap=cmap)

        # plt.title(title, fontsize='xx-large')
        if vmin is None:
            if vmax is None:
                vmax = max([hdfs[ff].max().max() for ff in feature_extraction_methods])

                plt.clim(vmin=0.0, vmax=vmax)
            else:
                plt.clim(vmin=0.0, vmax=vmax)
        else:
            if vmax is None:
                vmax = max([hdfs[ff].max().max() for ff in feature_extraction_methods])
                plt.clim(vmin=vmin, vmax=vmax)
            else:
                plt.clim(vmin=vmin, vmax=vmax)

        cbar = plt.colorbar(aspect=20, fraction=.12, pad=.02)
        cbar.ax.tick_params(labelsize=12)

        tick_marks = np.arange(len(select))
        plt.xticks(tick_marks, select, rotation=90, fontsize=14)
        plt.yticks(tick_marks, select, fontsize=14)

        plt.tight_layout()
        plt.ylabel('Training set', fontsize=20)
        plt.xlabel('Test set', fontsize=20)

        if filename.count('%s') == 0:
            plt.savefig(os.path.join(h5_dir, filename), bbox_inches='tight')
        elif filename.count('%s') == 1:
            plt.savefig(os.path.join(h5_dir, filename % feature_extraction), bbox_inches='tight')
        elif filename.count('%s') == 2:
            plt.savefig(os.path.join(h5_dir, filename % (dataset_name, feature_extraction)), bbox_inches='tight')
        else:
            print "Invalid filename, saving as 'default.png'."
            plt.savefig('default.png', bbox_inches='tight')
