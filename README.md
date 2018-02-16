# bachpro

This is a simple convolutional neural network, that I built as a refresher in
preparation for my Bachelor's Thesis. It is written in TensorFlow with the Estimator API.

My Bachelor Project will involve Residual Neural Networks. For its
construction, I might use GitHub as well, but the data set will be classified.

-- wkvanderveen


How to run this network:
1)  Make sure that the required packages are installed (scipy, matplotlib, tensorflow, numpy)
2)  Run the network using "python3 cnn_mnist.py". Change hyperparameters in "config.py" as desired.
3)  Optional: open a terminal, go to the main directory, and enter "tensorboard --logdir ./tmp_model_data".
    Then go to "http://localhost:6006". You will need to install tensorboard for this (using pip).
4)  Optional: go to the folder named "out" to view the weights and output of the convolutional layers.
