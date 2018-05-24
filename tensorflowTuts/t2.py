#import tensorflow as tf
import os
from skimage import data
from skimage import transform 
from skimage.color import rgb2gray
import numpy as np
import tensorflow as tf

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels


def imageTransformation(images):
    images28 = [transform.resize(image, (28, 28)) for image in images]
    # Convert `images28` to an array
    images28 = np.array(images28)

    # Convert `images28` to grayscale
    images28 = rgb2gray(images28)
    return images28

    
def buildGraph():
    
    '''
    None implies any size

    '''
    x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
    y = tf.placeholder(dtype = tf.int32, shape = [None])

    # Flatten the input data
    images_flat = tf.contrib.layers.flatten(x)

    # Fully connected layer 
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Define a loss function
    # Get the cross entropy and get the average of the same. this is to be reduced
    # That is it should match the labels = [0,0..1, 0 ,00..] as opposed to [0.5,0.4..0,2.]
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
    # Define an optimizer 
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Convert logits to label indexes

    # This returns the index off the largest value , alogn which dimension is to figured out 
    correct_pred = tf.argmax(logits, 1)

    # Define an accuracy metric --- > Studied till here !!!!!!!!!!!!!!!!!!
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def main():
    ROOT_PATH = "/mathworks/home/abelbabu/pythonScripts/RNNExperiments/"
    train_data_directory = os.path.join(ROOT_PATH, "data/Training")
    test_data_directory = os.path.join(ROOT_PATH, "data/Testing")
    images, labels = load_data(train_data_directory)

    images = imageTransformation(images)

if __name__ == '__main__':
    main()
    