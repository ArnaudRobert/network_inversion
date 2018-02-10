import tensorflow as tf
sess = tf.Session()
new_saver = tf.train.import_meta_graph('state_art_save.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

W_conv1=tf.get_collection('W_conv1')
b_conv1 = tf.get_collection('b_conv1')
#h_conv1 = tf.get_collection('h_conv1')
#h_pool1 =tf.get_collection('h_pool1')
#W_conv2 = tf.get_collection('W_conv2')
#b_conv2 = tf.get_collection('b_conv2')
#h_conv2 = tf.get_collection('h_conv2')
#h_pool2 = tf.get_collection('h_pool2')
#W_fc1 = tf.get_collection('W_fc1')
#b_fc1 = tf.get_collection('b_fc1')
#h_pool2_flat = tf.get_collection('h_pool2')
#h_fc1 = tf.get_collection('h_fc1')
#h_fc1_drop = tf.get_collection('h_fc1_drop')
#W_fc2 = tf.get_collection('W_fc2')
#b_fc2 = tf.get_collection('b_fc2')


W_conv1=sess.run(W_conv1)[0]
b_conv1=sess.run(b_conv1)[0]

print(W_conv1)
print(b_conv1)

#h_conv1=sess.run(h_conv1)[0]
#h_pool1 =sess.run(h_pool1)[0]
#W_conv2 = sess.run(W_conv2)[0]
#b_conv2 = sess.run(b_conv2)[0]
#h_conv2 = sess.run(h_conv2)[0]
#h_pool2 = sess.run(h_pool2)[0]
#W_fc1 = sess.run(W_fc1)[0]
#b_fc1 = sess.run(b_fc1)[0]
#h_pool2_flat = sess.run(h_pool2)[0]
#h_fc1 = sess.run(h_fc1)[0]
#h_fc1_drop = sess.run(h_fc1_drop)[0]
#W_fc2 = sess.run(W_fc2)[0]
#b_fc2 = sess.run(b_fc2)[0]