import datasets
from algorithm import *

m = 2

# train_points, train_labels, test_points, test_labels = datasets.subset(datasets.cifar10(), 5000, 1000)
# train_points, train_labels, test_points, test_labels = datasets.subset(datasets.cifarX(2), 5000, 1000)
train_points, train_labels, test_points, test_labels = datasets.subset(datasets.mnist10(), 5000, 1000)
print('Loaded input')
# print('Raw nearest-neighbor accuracy:', nn_accuracy(train_points, train_labels, test_points, test_labels))

train_points_2, test_points_2, graphs = algorithm(
    train_points,
    m=m,
    test_points=test_points,
    pca_dim=100,
    loss_weights=(1, 1, 1),
)
print(nn_accuracy(train_points_2, train_labels, test_points_2, test_labels))

plt_plot(train_points_2, train_labels)
plt.plot(graphs)
plt.show()


# train_points_1, test_points_1 = pca(train_points, m, test_points)
# print('PCA nearest-neighbor accuracy:', nn_accuracy(train_points_1, train_labels, test_points_1, test_labels))
#
# for i in range(5):
#     train_points_2, test_points_2, graphs = algorithm(
#         train_points,
#         m=m,
#         test_points=test_points,
#         pca_dim=100,
#         loss_weights=(1, 1, 1),
#         intermediate_dims=(10,),
#     )
#     with open('results/tenc_mytest.txt', 'a') as f:
#         print(nn_accuracy(train_points_2, train_labels, test_points_2, test_labels), end=',', file=f)
#
# for i in range(5):
#     train_points_2, test_points_2, graphs = algorithm(
#         train_points,
#         m=m,
#         test_points=test_points,
#         pca_dim=100,
#         loss_weights=(1, 0, 1),
#         intermediate_dims=(10,),
#     )
#     with open('results/tsne_mytest.txt', 'a') as f:
#         print(nn_accuracy(train_points_2, train_labels, test_points_2, test_labels), end=',', file=f)
#
# for i in range(5):
#     train_points_2, test_points_2, graphs = algorithm(
#         train_points,
#         m=m,
#         test_points=test_points,
#         pca_dim=100,
#         loss_weights=(0, 1, 1),
#         intermediate_dims=(10,),
#     )
#     with open('results/auto_mytest.txt', 'a') as f:
#         print(nn_accuracy(train_points_2, train_labels, test_points_2, test_labels), end=',', file=f)
