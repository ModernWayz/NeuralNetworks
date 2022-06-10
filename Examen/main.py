#
# Instructies: bij deze opdracht zie je in de data folder een audio MNIST dataset. Er zijn
# in die folder 2 subfolders: audio en audio-images. 'audio' bevat de WAV audio files van
# gesproken cijfers van 0-9. 'audio-images' bevat een spectrogram weergave van dezelfde audio
# files. Een spectrogram toont de verdeling van audio frequenties op elk tijdstip.
#
#
# Kortom: Waarbij MNIST dus afbeelding --> 0-9 labels bevat, bevat deze dataset audio --> 0-9 labels.
#
# Jouw taak is om een classifier te schrijven die leert om WAV files te classificeren tot het overeen-
# komstige cijfer 0-9. Je mag kiezen of je hierbij vertrekt van de WAV bestanden (gebruik helper functie
# get_wav_info) of van de spectrogrammen.
#

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import wave
import pylab
import random

# Set paths to input and output data
DIR = 'data/audio-images'

# Utility function to get sound and frame rate info
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# TODO Read and prepare dataset
# TODO Classify spoken words to digits 0-9
# Simply an example to get the Python file to run
# train_dataset = TODO
# valid_dataset = TODO
# model = TODO
# history = model.fit(TODO)

train_x, train_y, test_x, test_y = [], [], [], []

count = 0
for folder in random.sample(os.listdir(DIR), len(os.listdir(DIR))):
    if folder.startswith('class'):
        for file in os.listdir(DIR + '/' + folder):
            # Check later mss trainen we niet genoeg van 1 getal
            #wav_info = get_wav_info(DIR + '/' + folder + '/' + file)
            image = cv2.resize(cv2.imread(DIR + '/' + folder + '/' + file, 0), (200, 200))
            label = file.split('_')[0]
            if(count >= 2000):
                test_x.append(image)
                test_y.append(label)
            else:
                train_x.append(image)
                train_y.append(label)
            count += 1

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_x = train_x.reshape((-1, 200, 200, 1)).astype('float32') / 255
test_x = test_x.reshape((-1, 200, 200, 1)).astype('float32') / 255

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

history = model.fit(train_x, train_y, epochs = 10, batch_size = 128)

# indien plot niet werkt hier is het object
print(history.history)

# Dit was mijn persoonlijke history object
# {'loss': [2.3097851276397705, 1.8747541904449463, 1.58992338180542, 1.2933800220489502, 1.0019011497497559, 0.6465241312980652, 0.5165888071060181, 0.3525698184967041, 0.26019448041915894, 0.20629172027111053], 'accuracy': [0.1720000058412552, 0.27399998903274536, 0.40849998593330383, 0.5425000190734863, 0.652999997138977, 0.8040000200271606, 0.8330000042915344, 0.8974999785423279, 0.9315000176429749, 0.9484999775886536]}

# Plot the loss curves for training and validation.
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['loss'] # fout met plot dus dit is hetzelfde als erboven
epochs = range(1, len(loss_values)+1)

plt.figure(figsize=(8,6))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the accuracy curves for training and validation.
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(acc_values)+1)

plt.figure(figsize=(8,6))
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()