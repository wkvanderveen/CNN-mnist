import os
from shutil import rmtree

# General settings
overwrite_existing_model = True
model_dir = './tmp_model_data'
plot_dir = './out/plots'
batch_size = 50
num_epochs_train = 5
num_epochs_eval = 1
steps = 100
logging_interval = 100
verbose = True
plot_conv_weights = True
plot_conv_output = True
predict_afterwards = True
num_hidden = 1024
dropout_rate = 0.4
learning_rate = 0.01


def rem_existing_model():
    if overwrite_existing_model and os.path.exists(model_dir):
        rmtree(model_dir)
