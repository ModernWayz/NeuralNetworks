# François Chollet - Deep Learning with Python (2018, Manning)
# Example section 2.1 'A first look at a neural network' p41
# =================================================================

# You don’t need to try to reproduce this example on your machine just now. 
# If you wish to, you’ll first need to set up Keras, which is covered in section 3.3.
# The MNIST dataset comes preloaded in Keras, in the form of a set of four Numpy arrays.

## Listing 2.1 Loading the MNIST dataset in Keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# train_images and train_labels form the training set, the data that the model will learn from. 
# The model will then be tested on the test set, test_images and test_labels.
# The images are encoded as Numpy arrays, and the labels are an array of digits, ranging from 0 to 9. 
# The images and labels have a one-to-one correspondence.

# Let’s look at the training data:
print("Training Data:")
print(train_images.shape) # should be: (60000, 28, 28)
print(len(train_labels)) # should be: 60000
print(train_labels) # should be: array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)

# And here's the test data:
print("Test Data:")
print(test_images.shape) # should be: (10000, 28, 28)
print(len(test_labels)) # should be: 10000
print(test_labels) # should be: array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)

# The workflow will be as follows: First, we’ll feed the neural network the training data, train_images and train_labels. 
# The network will then learn to associate images and labels. Finally, we’ll ask the network to produce 
# predictions for test_images, and we’ll verify whether these predictions match the labels from test_labels.
# Let’s build the network—again, remember that you aren’t expected to understand everything about this example yet.

## Listing 2.2 The network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# The core building block of neural networks is the layer, a data-processing module that you can think of 
# as a filter for data. Some data goes in, and it comes out in a more use- ful form. 
# Specifically, layers extract representations out of the data fed into them—hopefully, 
# representations that are more meaningful for the problem at hand. Most of deep learning 
# consists of chaining together simple layers that will implement a form of progressive data distillation. 
# A deep-learning model is like a sieve for data process- ing, made of a succession of 
# increasingly refined data filters—the layers.
# Here, our network consists of a sequence of two Dense layers, which are densely connected (also called fully connected) 
# neural layers. The second (and last) layer is a 10-way softmax layer, which means it will return an 
# array of 10 probability scores (sum- ming to 1). Each score will be the probability that the current digit image 
# belongs to one of our 10 digit classes.
#
# To make the network ready for training, we need to pick three more things, as part
# of the compilation step:
#
#   A loss function: How the network will be able to measure its performance on the training data, 
#   and thus how it will be able to steer itself in the right direction.
#
#   An optimizer: The mechanism through which the network will update 
#   itself based on the data it sees and its loss function.
#
#   Metrics to monitor during training and testing: Here, we’ll only care about accuracy 
#   (the fraction of the images that were correctly classified).
#
# The exact purpose of the loss function and the optimizer will be made clear through- out the next two chapters.

## Listing 2.3 The compilation step
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Before training, we’ll preprocess the data by reshaping it into the shape the network expects 
# and scaling it so that all values are in the [0, 1] interval. 
# Previously, our train- ing images, for instance, were stored in an array of shape (60000, 28, 28) 
# of type uint8 with values in the [0, 255] interval. We transform it into a float32 array of shape (60000, 28 * 28) 
# with values between 0 and 1.

## Listing 2.4 Preparing the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# We also need to categorically encode the labels, a step that’s explained in chapter 3.

## Listing 2.5 Preparing the labels
### from keras.utils import to_categorical # Deze line werkt niet -> hieronder wel
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# We’re now ready to train the network, which in Keras is done via a call
# to the network’s fit method—we fit the model to its training data:

network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Two quantities are displayed during training: the loss of the network over the training data, 
# and the accuracy of the network over the training data.
# We quickly reach an accuracy of 0.989 (98.9%) on the training data. 
# Now let’s check that the model performs well on the test set, too:

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc) # should be: test_acc: 0.9785

# The test-set accuracy turns out to be 97.8%—that’s quite a bit lower than the training set accuracy. 
# This gap between training accuracy and test accuracy is an example of overfitting: 
# the fact that machine-learning models tend to perform worse on new data than on their training data. 
# Overfitting is a central topic in chapter 3.
# This concludes our first example—you just saw how you can build and train a neural network to classify 
# handwritten digits in less than 20 lines of Python code. In the next chapter, I’ll go into detail about 
# every moving piece we just previewed and clarify what’s going on behind the scenes. 
# You’ll learn about tensors, the data-storing objects going into the network; tensor operations, 
# which layers are made of; and gradient descent, which allows your network to learn from its training examples.