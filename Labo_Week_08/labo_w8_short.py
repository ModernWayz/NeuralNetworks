import os
import random
import cv2
import numpy as np
from keras import models, layers

train_images, train_labels, test_images, test_labels = [], [], [], []

for file in random.sample(os.listdir("Labo_Week_08/data/train/"), len(os.listdir("Labo_Week_08/data/train/"))):
    if file.endswith(".jpg") or file.endswith(".png"):
        label = 0 if file.startswith("cat") else 1 if file.startswith("dog") else None
        if label == None:
            continue
        image = cv2.resize(cv2.imread("Labo_Week_08/data/train/" + file, 0), (50, 50))
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
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))

network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

network.compile(optimizer='adam',
              loss= 'binary_crossentropy',
              metrics=['accuracy'])


train_images = train_images.reshape((-1, 50, 50, 1)).astype('float32') / 255
test_images = test_images.reshape((-1, 50, 50, 1)).astype('float32') / 255

network.fit(train_images, train_labels, epochs = 10, batch_size = 32)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc: ', test_acc)