import cnn_mnist
import os
from shutil import rmtree

"""
This is the config file of the network.
This file defines the hyperparameters of the actual CNN,
runs it,
and takes care of visualizations and graph production.
"""

# General settings
overwrite_old_checkpoint = True
checkpoint_directory = './tmp_model_data'

# Hyperparameters of CNN
cnn_hyperparams = {
    'batch_size': 100,
    'num_epochs_train': 15,
    'num_epochs_eval': 10,
    'steps': 5000,
    'logging_interval': 100,
    'verbose': True,
    'plot_conv_weights': False,
    'predict_afterwards': False,
    'plot_conv_output': True,
    'model_dir': checkpoint_directory
}

# Optionally delete old checkpoint
if overwrite_old_checkpoint and os.path.exists(checkpoint_directory):
    rmtree(checkpoint_directory)


# Run the actual CNN
cnn_mnist.main(cnn_hyperparams)
