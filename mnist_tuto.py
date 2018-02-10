"""
Train a two layer neural network on MNIST data.
Save weight and bias tensors
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  
  # Create a multilayer model.
  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  # first layer dimensions
  in_dim = 784; out_dim = 500
  # first layer 
  W1 = weight_variable([in_dim, out_dim])
  b1 = bias_variable([out_dim])
  l1_preactivate = tf.matmul(x, W1) + b1
  hidden1 = tf.nn.relu(l1_preactivate, name='activation')

  # secod layer dimensions 
  in_dim = 500; out_dim = 10
  # second layer
  W2 = weight_variable([in_dim, out_dim])
  b2 = bias_variable([out_dim])
  l2_preactivate = tf.matmul(hidden1, W2) + b2
  y = tf.identity(l2_preactivate, name='activation')

  keep_prob = tf.placeholder(tf.float32) 
   
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  tf.summary.scalar('cross_entropy', cross_entropy)
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
  
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      c_entropy = sess.run(cross_entropy, feed_dict=feed_dict(True))
      test_writer.add_summary(summary, i)
      print('{}) cross entropy : {}     accuracy : {}'.format(i, c_entropy,  acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()

  """ 
  # end of train save the model
  saver = tf.train.Saver({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})
  save_path = saver.save(sess, "./tmp/model.ckpt")
  saver.restore(sess, "./tmp/model.ckpt")
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


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=5000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default="./summaries",
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



# adding drop out stabilize training loss function hence weights -> to do so set default drop out to
# 0.9 
