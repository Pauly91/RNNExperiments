# Reference: https://gist.github.com/mikalv/3947ccf21366669ac06a01f39d7cff05

import tensorflow as tf
import numpy as np

#set hyperparameters
max_len = 40
step = 2
num_units = 128
learning_rate = 0.001
batch_size = 200
epoch = 60
temperature = 0.5

def read_data(file_name):
    '''
     open and read text file
    '''
    text = open(file_name, 'r').read()
    return text.lower()

def featurize(text):
    '''
     featurize the text to train and target dataset
    '''
    unique_chars = list(set(text))
    len_unique_chars = len(unique_chars)

    input_chars = []
    output_char = []


    # Try to fiddle with this to get other combintations.
    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])


    # a = np.zeros((4,3,2)) creates row of length 4 with 
    # Matrices of size 3*2
    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    target_data = np.zeros((len(input_chars), len_unique_chars))


    # The for loop give the data used for training the netwokr.
    # train_data is an np array of size len(input_chars) with 
    # each element of the row being max_len*len_unique_chars 
    # metrics, here max_len is chosen to be 40 and the unique
    # characters are length is 56. So essentially we are making
    # a one hot matrix for a sentence of length of 40, Here 40 is 
    # the row size so each row is one hot vector for a single character.
    # target_data is np array of size len(input_chars) with each
    # element a row of size len_unique _char, i.e one hot vector
    # for the output.

    # There should be a better way of representing other than 1 hot 
    # vectors - WordVec way of doing things ?


    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

def rnn(x, weight, bias, len_unique_chars):
    '''
     define rnn cell and prediction
    '''

    # Perm defines the order in which the axis are arranged, i.e [x,y,z] = [1,0,2] = > [y,x,z] 
    #  
    x = tf.transpose(x, [1, 0, 2]) # 2nd argument is perm:permutation 
    x = tf.reshape(x, [-1, len_unique_chars]) # -1 here refers to the none dimension in intial data
    x = tf.split(x, max_len, 0) # This split the data into sub-tensor of size max_len along x axis.


    '''
    Blogs to read in BasicLSTMCell
    - https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
    - https://medium.com/machine-learning-algorithms/build-basic-rnn-cell-with-static-rnn-707f41d31ee1


    '''

    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction

def sample(predicted):
    '''
     helper function to sample an index from a probability array
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

def run(train_data, target_data, unique_chars, len_unique_chars):
    '''
     main run function
    '''
    '''
    Hyper parameters:
    max_len = 40
    step = 2
    num_units = 128
    learning_rate = 0.001
    batch_size = 200
    epoch = 60
    temperature = 0.5
    '''


    x = tf.placeholder("float", [None, max_len, len_unique_chars]) # [ * , 40, one_hot_vector_length]
    y = tf.placeholder("float", [None, len_unique_chars]) # [ *, one_hot_vector_length
    weight = tf.Variable(tf.random_normal([num_units, len_unique_chars])) # 128,one_hot_vector_length
    bias = tf.Variable(tf.random_normal([len_unique_chars])) # one_hot_vector_length

    prediction = rnn(x, weight, bias, len_unique_chars)

    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    num_batches = int(len(train_data)/batch_size)

    for i in range(epoch):
        print("----------- Epoch {0}/{1} -----------").format(i+1, epoch)
        count = 0
        for _ in range(num_batches):
            train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
            count += batch_size
            sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})

        #get on of training set as seed
        seed = train_batch[:1:]

        #to print the seed 40 characters
        seed_chars = ''
        for each in seed[0]:
                seed_chars += unique_chars[np.where(each == max(each))[0][0]]
        print("Seed:", seed_chars)

        #predict next 1000 characters
        '''         
        for i in range(1000):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print('Result:', seed_chars) 
        '''
    sess.close()


if __name__ == "__main__":
    #get data from https://s3.amazonaws.com/text-datasets/nietzsche.txt
    text = read_data('/mathworks/home/abelbabu/pythonScripts/RNNExperiments/tensorflowTuts/nietzsche.txt')
    train_data, target_data, unique_chars, len_unique_chars = featurize(text)
    run(train_data, target_data, unique_chars, len_unique_chars)