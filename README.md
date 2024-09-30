# BYU-IGEM-2024

Using genetically modified plants as biosensors to measure chemicals in soil.

We then use machine learning in CNNs and linear regression to return the amount of chemical in the soil from an image of a biosensor plant.

This repo will be called in our app.

**main.py**
Runs inference on an image of a biosensor plant and returns chemical value.

**convolutional_neural_network.py**
Trains a convolutional neural network to identify images

**model.py**
Trains a linear regression model to predict chemical values of soil from images of plants.

**redify.py**
Redify, returns the average red pixel value of a image.
process image returns a csv with chemical labels and red values to train regression model on.

