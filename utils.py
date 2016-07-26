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

    
