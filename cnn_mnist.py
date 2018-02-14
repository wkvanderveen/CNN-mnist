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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.INFO)

PLOT_DIR = './out/plots'


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

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

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
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
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)),
                    bbox_inches='tight')


def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

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
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)),
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
                            units=1024,
                            activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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

    if (config["verbose"]):
        print("Loading data...")

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    if (config["verbose"]):
        print("\nCreating Estimator...\n")

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir=config["model_dir"])

    if (config["verbose"]):
        print("\nSet up logging for predictions...\n")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, at_end=True)

    if (config["verbose"]):
        print("\nTraining model...\n")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs_train'],
        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn,
                           steps=config['steps'],
                           hooks=[logging_hook])

    if (config["verbose"]):
        print("\nEvaluating model...\n")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=config['num_epochs_eval'],
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    """ https://github.com/grishasergei/conviz/blob/master/conviz.py
    """
    conv1_w = mnist_classifier.get_variable_value("Conv1/kernel")
    conv2_w = mnist_classifier.get_variable_value("Conv2/kernel")
    if config["plot_conv_weights"]:
        plot_conv_weights(conv1_w, 'conv1')
        plot_conv_weights(conv2_w, 'conv2')

    if config["predict_afterwards"]:
        # Evaluate the model and print results

        plot_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": mnist.test.images[:1]},
            batch_size=1,
            shuffle=False)

        for single_predict in mnist_classifier.predict(plot_input_fn):
            print(single_predict)
            print("Predicted class = {}".format(single_predict['classes']))
            print("Probablity = {}\n".format(single_predict['probabilities']))

    if config["plot_conv_output"]:
        # Plot the convolution outputs for a single input
        single_image = np.reshape(mnist.test.images[2], (28, 28))

        conv1_w = np.reshape(a=conv1_w, newshape=(32, 5, 5))
        conv2_w = np.reshape(a=conv2_w[:, :, 7, :], newshape=(32, 5, 5))

        convolutions1 = np.zeros((32, 24, 24))
        convolutions2 = np.zeros((32, 24, 24))

        for i in range(len(conv1_w)):
            convolutions1[i, :, :] = scipy.signal.convolve2d(conv1_w[i, :, :],
                                                            single_image,
                                                            mode="valid")

        for i in range(len(conv2_w)):
            convolutions2[i, :, :] = scipy.signal.convolve2d(conv2_w[i, :, :],
                                                            single_image,
                                                            mode="valid")

        convolutions1 = np.reshape(convolutions1, (1, 32, 24, 24))
        convolutions1 = np.swapaxes(convolutions1, 1, 2)
        convolutions1 = np.swapaxes(convolutions1, 2, 3)

        convolutions2 = np.reshape(convolutions2, (1, 32, 24, 24))
        convolutions2 = np.swapaxes(convolutions2, 1, 2)
        convolutions2 = np.swapaxes(convolutions2, 2, 3)

        plot_conv_output(convolutions1, 'conv1_output')
        plot_conv_output(convolutions2, 'conv2_output')


if __name__ == "__main__":
    tf.app.run()
