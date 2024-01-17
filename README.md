# TetrisAI.py

Disclaimer:

The Project is still not finished. It is work in progress

### How to use:
- Install requirements:
  ``pip install -r requirements. txt``
- Run main.py:
  ``python3 main.py``

This Python script is designed to train a Convolutional Neural Network (CNN) to play the game of Tetris on a NES emulator. The script uses the TensorFlow library to build and train the model, and the PyAutoGUI library to interact with the game.

The script begins by importing the necessary libraries and creating an empty list for the training data (`X_train`) and the labels (`y_train`). 

The model is then defined using TensorFlow's Keras API. It's a sequential model with two convolutional layers (each followed by a max pooling layer), a flattening layer, and two dense layers. The input shape of the first layer is set to match the dimensions of the game screen (240x256 pixels, with 3 color channels), and the output layer has 7 neurons, corresponding to the 7 possible actions (up, down, left, right, and three buttons on the NES controller).

The script then enters a loop where it captures screenshots of the game screen and processes them into a format suitable for the neural network. It also checks which key is currently being pressed and assigns a label to the current game state based on this. This data is then added to the training set.

Once the training data has been collected, it's converted into a numpy array and used to train the model. The model is trained using the Adam optimizer and the sparse categorical crossentropy loss function, which is suitable for multi-class classification problems like this one.

After training, the model's weights are saved to a file so that they can be loaded later without having to retrain the model.

The script then starts the game and enters a loop where it continuously captures the current game state, uses the trained model to predict the best action to take, and then performs this action. This loop continues until the game is over.

The `range` class provided is a built-in Python class and is not directly used in the main script. It's a sequence type that represents an immutable sequence of numbers and is commonly used for looping a specific number of times in for loops.
