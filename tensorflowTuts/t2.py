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

    
def NNBuildNRun(trainImages,trainLabels,testImages,testLabels):
    
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
    # sparse_softmax_cross_entropy_with_logits, label is an array of equal to the number of data
    # pointa with each value is a class number.

    # Define an optimizer 
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Convert logits to label indexes

    # This returns the index off the largest value , alogn which dimension is to figured out 
    correct_pred = tf.argmax(logits, 1)

    # Define an accuracy metric --- > This does not make sense, take the index and then
    # take the mean of it ?
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.set_random_seed(1234)
    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", correct_pred)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            f, loss_value = sess.run([train_op, loss], feed_dict={x: trainImages, y: trainLabels})

            print("Loss: ", loss)
            print("f: ",f)
        predicted = sess.run([correct_pred], feed_dict={x: testImages})[0]
        # Calculate correct matches 
        match_count = sum([int(y == y_) for y, y_ in zip(testLabels, predicted)])

        # Calculate the accuracy
        accuracy = match_count / len(testLabels)

        # Print the accuracy
        print("Accuracy: {:.3f}".format(accuracy))

            # Read this for debugging: https://wookayin.github.io/tensorflow-talk-debugging/#1


def main():
    ROOT_PATH = "/mathworks/home/abelbabu/pythonScripts/RNNExperiments/"
    train_data_directory = os.path.join(ROOT_PATH, "data/Training")
    test_data_directory = os.path.join(ROOT_PATH, "data/Testing")

    trainImages, trainLabels = load_data(train_data_directory)
    testImages, testLabels = load_data(test_data_directory)


    trainImages = imageTransformation(trainImages)
    testImages = imageTransformation(testImages)
    NNBuildNRun(trainImages,trainLabels,testImages,testLabels)

if __name__ == '__main__':
    main()
    