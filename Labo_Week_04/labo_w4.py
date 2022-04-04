# Imports
import os
import random
import cv2
import numpy as np
from keras import models, layers

# Set directory of train folder
DIR_TRAIN = "Labo_Week_04/data/train/"
# Set the amount of train data in each category -> the rest is test data
AMOUNT_SPLIT = 10000
# Set all images width and height (in px)
IMAGE_SIZE = 20
# Set values
EPOCHS = 7
BATCH_SIZE = 128

# Declare empty train & test arrays
train_images, train_labels, test_images, test_labels = [], [], [], []

# Loop through every file found in train folder and randomise the list with random.sample()
for file in random.sample(os.listdir(DIR_TRAIN), len(os.listdir(DIR_TRAIN))):
    # Check if the file is a valid image
    if file.endswith(".jpg") or file.endswith(".png"):
        # Check if the image is a dog or cat: 0 = cat, 1 = dog
        label = 0 if file.startswith("cat") else 1 if file.startswith("dog") else None
        # If label is not found -> cancel image
        if label == None:
            continue
        # Read and resize image file: 0 = grayscale, 1 = color
        image = cv2.resize(cv2.imread(DIR_TRAIN + file, 0), (IMAGE_SIZE, IMAGE_SIZE))
        # Check if the image is train or test data
        if(int(file.split(".")[1]) >= AMOUNT_SPLIT):
            # Add image and result to their respective array
            test_images.append(image)
            test_labels.append(label)
        else:
            # Add image and result to their respective array
            train_images.append(image)
            train_labels.append(label)

# Convert arrays to a numpy array
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(IMAGE_SIZE * IMAGE_SIZE,)))
# Set the last layer activation to sigmoid because it is the proper activation function for binary data
network.add(layers.Dense(1, activation='sigmoid'))
# Flatten the layers
network.add(layers.Flatten())
# Compile network: Set the loss to binary because we are working with 0 and 1 values
network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the image data: reshape with the length of the array and convert to float32
train_images = train_images.reshape((len(train_images), IMAGE_SIZE * IMAGE_SIZE)).astype('float32') / 255
test_images = test_images.reshape((len(test_images), IMAGE_SIZE * IMAGE_SIZE)).astype('float32') / 255

# Train the network
network.fit(train_images, train_labels, epochs = EPOCHS, batch_size = BATCH_SIZE)

# Check how it performs on test set
test_loss, test_acc = network.evaluate(test_images, test_labels)

# Print results
print('test_acc: ', test_acc)
#print('test_loss: ', test_loss)