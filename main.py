import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape the data to match the input shape of the model
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define the model using Input layer to avoid the warning
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Define input shape as the first layer
    layers.Flatten(),  # Flatten images to vectors (28x28 pixels -> 784)
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (for digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a directory to store TensorBoard logs
log_dir = "logs"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Create a custom function to log images based on user input
def log_images(epoch, logs):
    # Prompt user for the digit to view (0-9)
    digit_input = int(input("Enter a digit (0-9) to display in TensorBoard: "))
    
    # Validate user input
    if digit_input not in range(10):
        print("Invalid input! Please enter a digit between 0 and 9.")
        return
    
    # Find images of the selected digit in the test dataset
    digit_image = x_test[y_test == digit_input]  # Extract images where the label matches the user input
    
    # Choose the first image of the selected digit
    img = digit_image[0]
    
    # Log the image to TensorBoard
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.image(f"Digit {digit_input} Example", np.expand_dims(img, axis=0), step=epoch)

# Train the model with TensorBoard callback
model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=[tensorboard_callback, tf.keras.callbacks.LambdaCallback(on_epoch_end=log_images)])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
