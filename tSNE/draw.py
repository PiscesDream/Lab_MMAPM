print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold
import cPickle
from sklearn.decomposition import PCA

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                color=plt.cm.Set1(Y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def pca_visualize(X, Y, title, filename):
    print("Computing PCA embedding for {}".format(filename))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plot_embedding(X_pca, Y, title)
    plt.savefig(filename)
#

def visualize(X, Y, title, filename):
    print("Computing t-SNE embedding for {}".format(filename))
    tsne = manifold.TSNE(n_components=2, init='pca')#, random_state=0)
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne, Y, title)
    plt.savefig(filename)
#    plt.show()

def demo():
    from sklearn import datasets
    digits = datasets.load_digits(n_class=6)
    X = digits.data
    y = digits.target
    visualize(X, y)

if __name__ == '__main__':
    pass



