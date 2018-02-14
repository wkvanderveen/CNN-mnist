from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
import utils
import os
import config as cf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir_w = os.path.join(cf.plot_dir, 'conv_weights')
    plot_dir_w = os.path.join(plot_dir_w, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir_w, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # iterate channels
    for channel in channels:
        if cf.verbose:
            print("Plotting weight... Channel {0} of {1}...".format(channel+1, len(channels)))
        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))

        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img,
                      vmin=w_min,
                      vmax=w_max,
                      interpolation='nearest',
                      cmap='seismic')

            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir_w, '{}-{}.png'.format(name, str(channel))),
                    bbox_inches='tight')
        plt.close('all')


def plot_conv_output(conv_img, name, fig, axes, filename=None):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir_y = os.path.join(cf.plot_dir, 'conv_output')
    plot_dir_y = os.path.join(plot_dir_y, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir_y, empty=False)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img,
                  vmin=w_min,
                  vmax=w_max,
                  interpolation='bicubic',
                  cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    if filename is not None:
        plt.savefig(os.path.join(plot_dir_y, '{}_{}.png'.format(name, filename)),
                    bbox_inches='tight')
    else:
        plt.savefig(os.path.join(plot_dir_y, '{}.png'.format(name)),
                    bbox_inches='tight')

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(name="Conv1",
                             inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(name="Conv2",
                             inputs=pool1,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 32])
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=cf.num_hidden,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=cf.dropout_rate,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=cf.learning_rate)

        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main(config):

    if cf.overwrite_existing_model:
        if (cf.verbose):
            print("\nRemoving old model in {}...\n".format(cf.model_dir))
        cf.rem_existing_model()

    if (cf.verbose):
        print("\nLoading data...\n")

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    if (cf.verbose):
        print("\nCreating Estimator...\n")

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir=cf.model_dir)

    if (cf.verbose):
        print("\nSet up logging for predictions...\n")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, at_end=True)

    if (cf.verbose):
        print("\nTraining model...\n")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=cf.batch_size,
        num_epochs=cf.num_epochs_train,
        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn,
                           steps=cf.steps,
                           hooks=[logging_hook])

    if (cf.verbose):
        print("\nEvaluating model...\n")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=cf.num_epochs_eval,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    conv1_w = mnist_classifier.get_variable_value("Conv1/kernel")
    conv2_w = mnist_classifier.get_variable_value("Conv2/kernel")
    if cf.plot_conv_weights:
        if cf.verbose:
            print("\nPlotting the weights of the convolutional filters...\n")

        plot_conv_weights(conv1_w, 'conv1')
        plot_conv_weights(conv2_w, 'conv2')

    if cf.predict_afterwards:
        # Evaluate the model and print results
        if cf.verbose:
            print("\nPredicting one test instance...\n")

        plot_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": mnist.test.images[:1]},
            batch_size=1,
            shuffle=False)

        for single_predict in mnist_classifier.predict(plot_input_fn):
            print(single_predict)
            print("Predicted class = {}".format(single_predict['classes']))
            print("Probablity = {}\n".format(single_predict['probabilities']))

    if cf.plot_conv_output:
        if cf.verbose:
            print("\nPlotting the output of the convolutional filters...\n")

        single_image = np.reshape(mnist.test.images[1], (28, 28))

        # get number of convolutional filters
        num_filters = conv1_w.shape[3]

        # get number of grid rows and columns
        grid_r, grid_c = utils.get_grid_dim(num_filters)

        # create figure and axes
        fig, axes = plt.subplots(min([grid_r, grid_c]),
                                 max([grid_r, grid_c]))

        conv1_w = np.reshape(a=conv1_w, newshape=(32, 5, 5))

        convolutions = np.zeros((32, 24, 24))

        for i in range(len(conv1_w)):
            convolutions[i, :, :] = scipy.signal.convolve2d(conv1_w[i, :, :],
                                                            single_image,
                                                            mode="valid")

        convolutions = np.reshape(convolutions, (1, 32, 24, 24))
        convolutions = np.swapaxes(convolutions, 1, 2)
        convolutions = np.swapaxes(convolutions, 2, 3)
        plot_conv_output(convolutions, 'conv1_output', fig, axes)

        conv2_n_channels = conv2_w.shape[2]

        # get number of convolutional filters
        num_filters = conv2_w.shape[3]

        # get number of grid rows and columns
        grid_r, grid_c = utils.get_grid_dim(num_filters)

        for channel in range(conv2_n_channels):
            if cf.verbose:
                print("Plotting channel {0} of {1}\
                in convolution layer 2...".format(channel+1, conv2_n_channels))

            conv2_w_channel = np.reshape(a=conv2_w[:, :, channel, :],
                                         newshape=(32, 5, 5))
            convolutions = np.zeros((32, 24, 24))
            # create figure and axes
            fig, axes = plt.subplots(min([grid_r, grid_c]),
                                     max([grid_r, grid_c]))

            for i in range(len(conv2_w_channel)):
                convolutions[i, :, :] = scipy.signal.convolve2d(
                    conv2_w_channel[i, :, :], single_image, mode="valid")

            convolutions = np.reshape(convolutions, (1, 32, 24, 24))
            convolutions = np.swapaxes(convolutions, 1, 2)
            convolutions = np.swapaxes(convolutions, 2, 3)
            plot_conv_output(convolutions, 'conv2_output', fig, axes, filename=str(channel))
            plt.close('all')


if __name__ == "__main__":
    tf.app.run()
