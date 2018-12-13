from helper_functions import *


def algorithm(
        input_points,
        m=2,  # The number of dimensions to reduce to
        batch_size=512,
        iters=20000,
        test_points=None,  # Additional points to run through the dimensionality reduction after training
        pca_dim=250,
        perplexity=20,  # Approximately the number of neighbors of each point for T-SNE's gaussians
        network_depth=2,
        hidden_units=50,
        batch_norm=True,
        loss_weights=(1, 1, 1),  # How important each component of the loss is
        intermediate_dims=(),  # The dimensions the network should reduce the input to along the way
):
    # Ensure inputs are reasonable
    N, M = input_points.shape
    input_points = input_points.astype(np.float32)
    test_points = test_points.astype(np.float32)
    batch_size = min(N, batch_size)

    # Run principal component analysis
    if pca_dim is not None:
        print('Running PCA...')
        pca_mat = pca_matrix(input_points, pca_dim).astype(np.float32)
        input_points = np.dot(input_points, pca_mat)
        if test_points is not None:
            test_points = np.dot(test_points, pca_mat)
        M = pca_dim

    # Estimate the dataset's density near each point
    print('Estimating variances...')
    t_all = estimate_t(input_points, perplexity=perplexity).astype(np.float32)

    # Define the neural network used for embeddings
    def transform(input, ndims):
        x = input
        for _ in range(network_depth):
            x = tf.layers.dense(x, hidden_units, tf.nn.leaky_relu)
            if batch_norm:
                x = tf.layers.batch_normalization(x)
        return tf.layers.dense(x, ndims)

    # Define tensorflow placeholders
    batch = tf.placeholder(tf.int32, [None])
    exaggeration = tf.placeholder(tf.float32)
    l2_penalty = tf.placeholder(tf.float32)
    test_run = tf.placeholder(tf.bool)

    # Define the input data
    D = tf.cond(test_run,
                lambda: test_points,
                lambda: tf.gather(input_points, batch))
    t = tf.gather(t_all, batch)

    # Define all the embeddings
    X1 = [D]
    for d in intermediate_dims:
        X1.append(transform(X1[-1], d))
    Y = transform(X1[-1], m)
    X2 = [Y]
    for d in intermediate_dims[::-1]:
        X2.append(transform(X2[-1], d))
    D2 = transform(X2[-1], M)

    # Define the components of the loss
    loss1 = tsne_loss(D, Y, t, batch_size, exaggeration)
    loss2 = tf.reduce_mean(tf.square(D - D2))
    loss3 = l2_penalty * tf.reduce_mean(Y * Y)
    for x1, x2 in zip(X1[1:], X2[1:]):
        loss1 += tsne_loss(D, x1, t, batch_size, exaggeration)
        loss2 += tf.reduce_mean(tf.square(x1 - x2))

    # Define the loss and training optimizer
    loss_components = [loss1, loss2, loss3]
    loss = tf.reduce_sum(tf.stack(loss_components) * loss_weights)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # Setup tensorflow session for training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        graphs = []

        # Iterate through a large number of steps
        for i in range(1, iters + 1):
            # Pick a batch uniformly at random
            random_batch = np.random.choice(N, size=[batch_size], replace=False)
            # Define training parameters for this step
            early_exag = 1  # + 10 * np.exp(i * -.01)
            early_l2p = np.exp(i * -.001)
            # Map each placeholder to the right value
            feed_dict = {batch: random_batch, exaggeration: early_exag, l2_penalty: early_l2p, test_run: False}

            # Run the optimizer for one step
            *loss_out, _ = sess.run(loss_components + [train_op], feed_dict=feed_dict)
            graphs.append(loss_out)
            # Print output periodically
            if i % 100 == 0:
                print('Step {}, loss {}'.format(i, loss_out))

        if test_points is None:
            # Return the reduced input points and the graph of the loss components
            return sess.run(Y, feed_dict={batch: np.arange(N), test_run: False}), \
                   graphs
        else:
            # Return the reduced input points, the reduced test points, and the graph of the loss components
            return sess.run(Y, feed_dict={batch: np.arange(N), test_run: False}), \
                   sess.run(Y, feed_dict={batch: np.arange(N), test_run: True}), \
                   graphs
