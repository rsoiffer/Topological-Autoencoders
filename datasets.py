import os
import tarfile
from urllib.request import urlretrieve

import mnist
import numpy as np


def cifarX(num_classes, path='datasets', onehot=False):
    input_points, input_labels, test_points, test_labels = cifar10(path, onehot)
    input_choice = input_labels < num_classes
    input_points, input_labels = input_points[input_choice, :], input_labels[input_choice]
    test_choice = test_labels < num_classes
    test_points, test_labels = test_points[test_choice, :], test_labels[test_choice]
    return input_points, input_labels, test_points, test_labels


def cifar10(path='datasets', onehot=False):
    # Code from https://mattpetersen.github.io/load-cifar10-with-numpy

    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser('~'), 'data', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    if onehot:
        return train_images, _onehot(train_labels), test_images, _onehot(test_labels)
    else:
        return train_images, train_labels, test_images, test_labels


def mnistX(num_classes):
    input_points, input_labels, test_points, test_labels = mnist10()
    input_choice = input_labels < num_classes
    input_points, input_labels = input_points[input_choice, :], input_labels[input_choice]
    test_choice = test_labels < num_classes
    test_points, test_labels = test_points[test_choice, :], test_labels[test_choice]
    return input_points, input_labels, test_points, test_labels


def mnist10():
    mndata = mnist.MNIST('datasets')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    return np.array(train_images) / 255, np.array(train_labels), np.array(test_images) / 255, np.array(test_labels)


def subset(dataset, train_size, test_size):
    train_points, train_labels, test_points, test_labels = dataset
    # train_choice = np.random.choice(len(train_labels), train_size, False)
    # test_choice = np.random.choice(len(test_labels), test_size, False)
    train_choice = np.arange(train_size)
    test_choice = np.arange(test_size)
    return train_points[train_choice, :], train_labels[train_choice], \
           test_points[test_choice, :], test_labels[test_choice]
