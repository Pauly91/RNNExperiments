
#Reference: https://www.datacamp.com/community/tutorials/tensorflow-tutorial


# Import `tensorflow`
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)
config1 = tf.ConfigProto(log_device_placement = True)

with tf.Session(config=config1) as sess:
    output = sess.run(result)
    print(output)
