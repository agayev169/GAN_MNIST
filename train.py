# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

D_input_n = 784
D_h1_n = 1024
D_h2_n = 512
D_h3_n = 256
D_output_n = 1

G_input_n = 100
G_h1_n = 256
G_h2_n = 512
G_h3_n = 1024
G_output_n = 784

batch_size = 128
learning_rate = 0.001
epochs_n = 1000

X = tf.placeholder(tf.float32, shape=(None, 784))
Z = tf.placeholder(tf.float32, shape=(None, 100))

def generator(X):
	G_h1 = tf.layers.dense(X, G_h1_n, activation=tf.nn.leaky_relu)
	G_h2 = tf.layers.dense(G_h1, G_h2_n, activation=tf.nn.leaky_relu)
	G_h3 = tf.layers.dense(G_h2, G_h3_n, activation=tf.nn.leaky_relu)
	G_output = tf.layers.dense(G_h3, G_output_n, activation=tf.nn.tanh)

	return G_output

def discriminator(X):
	D_h1 = tf.layers.dropout(
		tf.layers.dense(X, D_h1_n, activation=tf.nn.leaky_relu), 0.3)
	D_h2 = tf.layers.dropout(
		tf.layers.dense(D_h1, D_h2_n, activation=tf.nn.leaky_relu), 0.3)
	D_h3 = tf.layers.dropout(
		tf.layers.dense(D_h2, D_h3_n, activation=tf.nn.leaky_relu), 0.3)
	D_output = tf.layers.dense(D_h3, D_output_n, activation=tf.nn.sigmoid)

	return D_output

G_z = generator(Z)

D_real = discriminator(X)
D_fake = discriminator(G_z)

eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

D_opt = tf.train.AdamOptimizer(learning_rate).minimize(D_loss)
G_opt = tf.train.AdamOptimizer(learning_rate).minimize(G_loss)

init = tf.global_variables_initializer()

def mean(data):
	avg = 0
	for x in data:
		avg += x
	return avg / len(data)

saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
	sess.run(init)

	print("Start of a training")
	start_tr = time.time()
	for epoch in range(epochs_n):
		G_losses = []
		D_losses = []
		start_epoch = time.time()

		for _ in range(len(mnist.train.images) // batch_size):
			x, _ = mnist.train.next_batch(batch_size)
			z = np.random.normal(0, 1, (batch_size, 100))

			loss_d, _ = sess.run([D_loss, D_opt], {X: x, Z: z})
			D_losses.append(loss_d)

			z = np.random.normal(0, 1, (batch_size, 100))
			loss_g, _ = sess.run([G_loss, G_opt], {Z: z})
			G_losses.append(loss_g)

		end_epoch = time.time()
		print(epoch + 1, "epochs from", epochs_n)
		print("Discriminator loss:", mean(D_losses[-len(mnist.train.images) // batch_size:]))
		print("Generator loss:", mean(G_losses[-len(mnist.train.images) // batch_size:]))
		print("Time spent for", epoch + 1, "epoch:", end_epoch - start_epoch)

		if epoch % 10 == 0:
			save_path = saver.save(sess, "models2/model" + str(epoch + 1) + ".ckpt")
			print("Model saved in path: %s" % save_path)

		print("")

	end_tr = time.time()
	print("Discriminator loss:", mean(D_losses))
	print("Generator loss:", mean(G_losses))
	print("Time spent for training:", end_tr - start_tr)