import os
import random
import cv2
import numpy as np
from keras import models, layers

train_images, train_labels, test_images, test_labels = [], [], [], []

for file in random.sample(os.listdir("Labo_Week_04/data/train/"), len(os.listdir("Labo_Week_04/data/train/"))):
    if file.endswith(".jpg") or file.endswith(".png"):
        label = 0 if file.startswith("cat") else 1 if file.startswith("dog") else None
        if label == None:
            continue
        image = cv2.resize(cv2.imread("Labo_Week_04/data/train/" + file, 0), (20, 20))
        if(int(file.split(".")[1]) >= 10000):
            test_images.append(image)
            test_labels.append(label)
        else:
            train_images.append(image)
            train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(20 * 20,)))
network.add(layers.Dense(1, activation='sigmoid'))
network.add(layers.Flatten())
network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((len(train_images), 20 * 20)).astype('float32') / 255
test_images = test_images.reshape((len(test_images), 20 * 20)).astype('float32') / 255

network.fit(train_images, train_labels, epochs = 10, batch_size = 128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc: ', test_acc)