# Tetris played by AI
from nes import NES
import pyautogui as pag
import tensorflow as tf
import numpy as np
import cv2
import numpy as np

# Create a NES object

# Controls are:
# Up, Left, Down, Right: W, A, S, D
# Select, Start:  G, H
# A, B: P, L

# Create your own model
X_train = []  # Define X_train variable

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(240, 256, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])

y_train = []  # Define y_train variable

# Populate X_train and y_train with actual data
for i in range(1, 10):
    # Get the image of the game
    img = pag.screenshot(region=(0, 0, 256, 240))
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = img_np / 255.0
    X_train.append(img_np)

    # Get the label
    label = 0
    if pag.keyDown('w'):
        label = 1
    elif pag.keyDown('a'):
        label = 2
    elif pag.keyDown('s'):
        label = 3
    elif pag.keyDown('d'):
        label = 4
    elif pag.keyDown('p'):
        label = 5
    elif pag.keyDown('l'):
        label = 6
    y_train.append(label)

X_train = np.array(X_train)  # Convert training data to numpy array
y_train = np.array(y_train)  # Convert training data and labels to numpy arrays

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train your model
model.fit(X_train, y_train, epochs=10)

# Save the trained model
model.save_weights("tetris.h5")

# Start the game
nes = NES("tetris.nes")
nes.run()

# Use the trained model to play the game
while True:
    # Get the image of the game
    img = pag.screenshot(region=(0, 0, 256, 240))
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = img_np / 255.0

    # Predict the action using the trained model
    prediction = model.predict(np.expand_dims(img_np, axis=0))
    action = np.argmax(prediction)

    # Perform the action by pressing the corresponding key
    if action == 1:
        pag.press('w')
    elif action == 2:
        pag.press('a')
    elif action == 3:
        pag.press('s')
    elif action == 4:
        pag.press('d')
    elif action == 5:
        pag.press('p')
    elif action == 6:
        pag.press('l')

# Check if the game is over
    if check.is_game_over():
        break

