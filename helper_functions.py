import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def estimate_t(input_points, perplexity):
    n = input_points.shape[0]
    psd = np.square(squareform(pdist(input_points)))

    def entropy(t):
        P = np.exp(-psd / t) * (1 - np.eye(n))
        P = np.maximum(P / np.sum(P, 0), 1e-12)
        return -np.sum(P * np.log(P), 0)

    return find_zero_parallel(
        lambda y: entropy(y) - np.log(perplexity),
        lambda y: 10 * np.exp(3 * y), n)


def find_zero_parallel(f, g, n, iters=20):
    x = np.zeros([n])
    for i in range(1, iters):
        f_output = f(g(x))
        if i % 5 == 0:
            print('Finding zeros, iteration {}, mean error {}'.format(i, np.mean(np.abs(f_output))))
        x -= np.sign(f_output) * (2 ** -i)
    return g(x)


def nn_accuracy(points, labels, test_points=None, test_labels=None):
    print('Computing nearest-neighbor accuracy...')
    kdtree = scipy.spatial.cKDTree(points)
    correct = 0.0
    num_points = len(points if test_points is None else test_points)
    eval_points = points if test_points is None else test_points
    eval_labels = labels if test_labels is None else test_labels

    for i in range(num_points):
        p = eval_points[i]
        if test_points is None:
            n = kdtree.query(p, 2)[1][1]
        else:
            n = kdtree.query(p, 1)[1]
        correct += 1 if labels[n] == eval_labels[i] else 0
    print(correct, num_points)
    return correct / num_points


def pairwise_square_distances(a):
    r = tf.reshape(tf.reduce_sum(a * a, 1), [-1, 1])
    return r - 2 * tf.matmul(a, tf.transpose(a)) + tf.transpose(r)


def pca(input_points, m, test_points=None):
    pca_mat = pca_matrix(input_points, m)
    if test_points is None:
        return np.dot(input_points, pca_mat)
    else:
        return np.dot(input_points, pca_mat), np.dot(test_points, pca_mat)


def pca_matrix(input_points, m):
    x = input_points - np.mean(input_points, 0)
    covariance = np.cov(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, -m:]


def plt_plot(points, labels):
    m = points.shape[1]
    if m == 2:
        plt.scatter(points[:, 0], points[:, 1], c=labels)
    elif m == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels)
    plt.colorbar()
    plt.show()


def tsne_loss(D, Y, t, batch_size, exaggeration):
    psd_D = pairwise_square_distances(D)
    weights = tf.exp(-psd_D / t) * (1 - np.eye(batch_size))
    P = weights / tf.reduce_sum(weights, 0)
    P = P + tf.transpose(P)
    P = P / tf.reduce_sum(P)
    P = tf.maximum(P, 1e-12)
    P = P * exaggeration

    psd_Y = pairwise_square_distances(Y)
    tdist = 1 / (1 + psd_Y) * (1 - np.eye(batch_size))
    Q = tdist / tf.reduce_sum(tdist)
    Q = tf.maximum(Q, 1e-12)

    return tf.reduce_sum(P * tf.log(P / Q))
