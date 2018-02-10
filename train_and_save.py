"""
Train a two layer neural network on MNIST data.
Save weight and bias tensors
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



#########################################################
# Experiment parameters
#########################################################

seed = 123
n_iters = 100         # number of iterations for training
to_save = True        # if True save the learned weights and biases
keep_proba = 0.9      # probability for dropout 
lr = 1e-3             # Adam learning rate 

# TODO add various loss functions

# set seed for reproducibility
tf.set_random_seed(seed)
sess = tf.InteractiveSession()

#########################################################
# Helper functions 
#########################################################

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = keep_proba
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


#########################################################
# Build Network
#########################################################

x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.int64, [None, 10], name='y-input')

image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])


# first layer dimensions
in_dim = 784; out_dim = 500
# first layer 
W1 = weight_variable([in_dim, out_dim])
b1 = bias_variable([out_dim])
l1 = tf.matmul(x, W1) + b1
h1 = tf.nn.relu(l1, name='activation')

# secod layer dimensions 
in_dim = 500; out_dim = 10
# second layer
W2 = weight_variable([in_dim, out_dim])
b2 = bias_variable([out_dim])
l2 = tf.matmul(h1, W2) + b2
y = tf.identity(l2, name='activation')

keep_prob = tf.placeholder(tf.float32)
#cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

tf.summary.scalar('cross_entropy', cross_entropy)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(n_iters):
    if i % 10 == 0:  # Record summaries and test-set accuracy
        c_entropy = sess.run(cross_entropy, feed_dict=feed_dict(True))
        print('{}) cross entropy : {}'.format(i, c_entropy))
    # train step 
    _ = sess.run(train_step, feed_dict=feed_dict(True))


if to_save:
    # save the weights
    folder = "saved_weights/"
    np.save(folder+'W1', W1.eval())
    np.save(folder+'b1', b1.eval())
    np.save(folder+'W2', W2.eval())
    np.save(folder+'b2', b2.eval())


"""
  # now we fix the weight and make up the most suited input for class 0
  x_ = tf.weight_variable([1, 784])
  ys = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  in_dim = 784; out_dim = 500
  W1_ = tf.placeholder(tf.float32, shape=[in_dim, out_dim])
  b1_ = tf.placeholder(tf.float32, shape=[out_dim])
  l1_ = tf.matmul(x_, W1_) + b1_
  h1_ = tf.nn.relu(l1, name='activation')


  in_dim = 500; out_dim = 10
  W2_ = tf.placeholder(tf.float32, shape=[in_dim, out_dim])
  b2_ = tf.placeholder(tf.float32, shape=[out_dim])
  l2_ = tf.matmul(h1_, W2_) + b2_
  y_ = tf.identity(l2_, name='activation')

"""

