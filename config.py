import os
from shutil import rmtree

# General settings
overwrite_existing_model = False
model_dir = './tmp_model_data'
plot_dir = './plots'
batch_size = 200
num_epochs_train = 20
num_epochs_eval = 2
steps = 500
logging_interval = 100
verbose = True
plot_conv_weights = True
plot_conv_output = True
predict_afterwards = True
num_hidden = 1024
dropout_rate = 0.4
learning_rate = 0.0005


def rem_existing_model():
    """Delete map containing model metadata."""
    if overwrite_existing_model and os.path.exists(model_dir):
        rmtree(model_dir)
