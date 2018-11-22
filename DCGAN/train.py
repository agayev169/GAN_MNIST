import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../MNIST_data/")

G_input_n = (None, 1, 1, 100)
G_conv1_n = 1024
G_conv2_n = 512
G_conv3_n = 256
G_conv4_n = 128
G_output_n = 1

D_input_n = (None, 64, 64, 1)
D_conv1_n = 128
D_conv2_n = 256
D_conv3_n = 512
D_conv4_n = 1024
D_output_n = 1

batch_size = 128
learning_rate = 0.0001
epochs_n = 100

def generator(X, training=True):
	with tf.variable_scope("generator", reuse=False):
		G_conv1 = tf.layers.conv2d_transpose(X, G_conv1_n, [4, 4], strides=(1, 1), padding='valid')
		G_conv1 = tf.nn.relu(tf.layers.batch_normalization(G_conv1, training=training))

		G_conv2 = tf.layers.conv2d_transpose(G_conv1, G_conv2_n, [4, 4], strides=(2, 2), padding='same')
		G_conv2 = tf.nn.relu(tf.layers.batch_normalization(G_conv2, training=training))

		G_conv3 = tf.layers.conv2d_transpose(G_conv2, G_conv3_n, [4, 4], strides=(2, 2), padding='same')
		G_conv3 = tf.nn.relu(tf.layers.batch_normalization(G_conv3, training=training))

		G_conv4 = tf.layers.conv2d_transpose(G_conv3, G_conv4_n, [4, 4], strides=(2, 2), padding='same')
		G_conv4 = tf.nn.relu(tf.layers.batch_normalization(G_conv4, training=training))

		G_output = tf.layers.conv2d_transpose(G_conv4, G_output_n, [4, 4], strides=(2, 2), padding='same')
		G_output = tf.nn.sigmoid(G_output)

		return G_output

def discriminator(X, training, reuse=False):
	with tf.variable_scope("discriminator", reuse=reuse):
		D_conv1 = tf.layers.conv2d(X, D_conv1_n, [4, 4], strides=(2, 2), padding='same')
		D_conv1 = tf.nn.relu(tf.layers.batch_normalization(D_conv1, training=training))

		D_conv2 = tf.layers.conv2d(D_conv1, D_conv2_n, [4, 4], strides=(2, 2), padding='same')
		D_conv2 = tf.nn.relu(tf.layers.batch_normalization(D_conv2, training=training))

		D_conv3 = tf.layers.conv2d(D_conv2, D_conv3_n, [4, 4], strides=(2, 2), padding='same')
		D_conv3 = tf.nn.relu(tf.layers.batch_normalization(D_conv3, training=training))

		D_conv4 = tf.layers.conv2d(D_conv3, D_conv4_n, [4, 4], strides=(2, 2), padding='same')
		D_conv4 = tf.nn.relu(tf.layers.batch_normalization(D_conv4, training=training))

		D_output = tf.layers.conv2d(D_conv4, D_output_n, [4, 4], strides=(1, 1), padding='valid')
		D_output_s = tf.nn.sigmoid(D_output)

		return D_output, D_output_s


X = tf.placeholder(tf.float32, shape=D_input_n)
Z = tf.placeholder(tf.float32, shape=G_input_n)
training = tf.placeholder(dtype=tf.bool)

G_z = generator(Z, training)
D_real, D_real_logits = discriminator(X, training)
D_fake, D_fake_logits = discriminator(G_z, training, True)

D_loss_real = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if var.name.startswith('discriminator')]
G_vars = [var for var in t_vars if var.name.startswith('generator')]

D_opt = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)
G_opt = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=G_vars)

init = tf.global_variables_initializer()


def mean(data):
	avg = 0
	for x in data:
		avg += x
	return avg / len(data)


fixed_z = np.random.normal(0, 1, (25, 1, 1, 100))
def save_generated(rows_n, cols_n, epoch, fixed=True, path="imgs"):
	try:
		os.mkdir(path)
	except Exception as e:
		pass

	if fixed == True:
		imgs = sess.run(G_z, {Z: fixed_z})
	else:
		z = np.random.normal(0, 1, (rows_n * cols_n, 100))
		imgs = sess.run(G_z, {Z: z, training: False})
	plt.figure(1, figsize=(20, 20))
	for j in range(len(imgs)):
		plt.subplot(cols_n, rows_n, j + 1)
		plt.title("Img #" + str(j + 1))
		plt.imshow(np.reshape(imgs[j], [64, 64]), cmap='gray')

	plt.savefig(path + "/epoch-{}.png".format(epoch))   
	plt.close()

saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
	sess.run(init)

	print("Start of a training")
	start_tr = time.time()
	for epoch in range(epochs_n):
		G_losses = []
		D_losses = []
		start_epoch = time.time()

		save_generated(5, 5, epoch + 1, fixed=False)

		for _ in range(len(mnist.train.images) // batch_size):
			x, _ = mnist.train.next_batch(batch_size)
			x = tf.image.resize_images(x, [64, 64]).eval()
			z = np.random.normal(0, 1, (batch_size, 1, 1, 100))

			loss_d, _ = sess.run([D_loss, D_opt], {X: x, Z: z, training: True})
			D_losses.append(loss_d)

			z = np.random.normal(0, 1, (batch_size, 1, 1, 100))
			loss_g, _ = sess.run([G_loss, G_opt], {Z: z, training: True})
			G_losses.append(loss_g)

		end_epoch = time.time()
		print(epoch + 1, "epochs from", epochs_n)
		print("Discriminator loss:", mean(D_losses[-len(mnist.train.images) // batch_size:]))
		print("Generator loss:", mean(G_losses[-len(mnist.train.images) // batch_size:]))
		print("Time spent for epoch #{}: {}".format(epoch + 1, end_epoch - start_epoch))

		save_generated(5, 5, epoch + 1, fixed=False)

		save_path = saver.save(sess, "models/model" + str(epoch + 1) + ".ckpt")
		print("Model saved in path: %s" % save_path)

		print("")

	end_tr = time.time()
	print("Discriminator loss:", mean(D_losses))
	print("Generator loss:", mean(G_losses))
	print("Time spent for training:", end_tr - start_tr)