# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:10:29 2018

@author: Thomas Athey
"""

# Import tensorflow
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/MNIST_data/",
                                  one_hot=True,
                                  source_url='http://yann.lecun.com/exdb/mnist/')
# Load a training sample
training_sample_ind = 50
training_sample_image = mnist.train.images[training_sample_ind, :]
training_sample_label = mnist.train.labels[training_sample_ind, :]

# Inspect image and label shapes
print('One-hot encoded training label: {}'.format(', '.join(map(str, training_sample_label))))
print('Integer training label: {}'.format(np.argmax(training_sample_label)))
print('Flattened training image shape: {}'.format(str(training_sample_image.shape)))

# Display image
#training_sample_image_reshaped = np.reshape(training_sample_image, [28, 28])
#plt.imshow(training_sample_image_reshaped, cmap=plt.get_cmap('gray_r'))
#plt.show()

x = tf.placeholder(tf.float32, [None, 784])  # batch of images: (batch_size, 28*28)
y_ = tf.placeholder(tf.float32, [None, 10])  # batch of image labels: (batch_size, 10), one-hot encoded.


def conv2d(x, W):
    # x: [batch_size, in_height, in_width, in_channels]
    # W: [filter_height, filter_width, in_channels, out_channels]
    # strides: 1-D tensor of length 4. The stride of the sliding window for each dimension of input.
    # padding: 'SAME' (zero pad input to ensure output has the same height/width)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # x: [batch_size, in_height, in_width, in_channels]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Reshape input of [batch_size, 784] to [batch_size, 28, 28, 1]
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
# With probability keep_prob, outputs the input element scaled up by 1 / keep_prob,
# otherwise outputs 0. The scaling is so that the expected sum is unchanged.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Loss function (cross entropy)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# Training operation
learning_rate = 1e-4
optimizer_name = 'Adam'

if optimizer_name == 'sgd':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate)
elif optimizer_name == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
elif optimizer_name == 'Adagrad':
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
elif optimizer_name == 'Adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
elif optimizer_name == 'Adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
else:
    raise ValueError("Not supported optimizer: %s", optimizer_name)

train_step = optimizer.minimize(cross_entropy)


# Evaluation metric
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize variables and session

init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

sess = tf.Session()
sess.run(init)
sess.run(init_l)

# Run mini-batch training on 50 elements 20000 times.
nb_iterations = 20000
batch_size = 50
dropout_rate = 0.5


for i in range(nb_iterations):
    batch = mnist.train.next_batch(batch_size)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
             x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy)) 

    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


    
    