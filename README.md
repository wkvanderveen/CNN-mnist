# bachpro

This is a project I'm doing for my AI BSc.
The goal is to build a liver analysis system using a deep convolution neural network,
but right now I'm just testing some basic CNN in TensorFlow, and checking if everything
works with Git. I can use this initial system to bootstrap the project system.

Over the next couple of weeks/months, I'll continue to improve the system.

Future plans right now include adding a personal log file; adding TensorBoard
visualizations; and improving the system organization and code in general.

Future plans right now include building an automatic hyperparameter optimizer,
improving the GUI and performance visualization, and improving the legibility of the code.

-- wkvanderveen



----------
some notes for myself:
* open tensorboard in model tmp file with "--logdir ./tmp_model_data"
* http://localhost:6006/


How to run this network:
1)  Make sure that the required packages are installed (scipy, matplotlib, tensorflow, numpy)
2)  Run the network using "python3 cnn_mnist.py". Change hyperparameters in "config.py" as desired.
3)  Optional: open a terminal, go to the main directory, and enter "tensorboard --logdir ./tmp_model_data".
    Then go to "http://localhost:6006". You will need to install tensorboard for this (using pip).
4)  Optional: go to the folder named "out" to view the weights and output of the convolutional layers.
