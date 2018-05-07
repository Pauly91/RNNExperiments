# A simple Echo-RNN that remembers the input data and then echoes 
# it after a few time-steps.

# Referece: https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767


from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000 # Total length of the series
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3 # The time steps after which the output is given out by the RNN, ex in: 011011 out: 000011011, i.e. a shift of three 
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length # // is floored division.




def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)


def main():
    x,y = generateData()
    print('x:',x)
    print('y:',y)

    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

    init_state = tf.placeholder(tf.float32, [batch_size, state_size])   

    W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
    b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

    W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

if __name__ == '__main__':
    main()
