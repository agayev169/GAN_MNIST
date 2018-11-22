import matplotlib.pyplot as plt
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
learning_rate = 0.0002
epochs_n = 1000

def generator(X):
	w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
	b_init = tf.constant_initializer(0.)

	G_w0 = tf.get_variable('G_w0', [G_input_n, G_h1_n], initializer=w_init)
	G_b0 = tf.get_variable('G_b0', [G_h1_n], initializer=b_init)
	G_h0 = tf.nn.relu(tf.matmul(X, G_w0) + G_b0)

	G_w1 = tf.get_variable('G_w1', [G_h1_n, G_h2_n], initializer=w_init)
	G_b1 = tf.get_variable('G_b1', [G_h2_n], initializer=b_init)
	G_h1 = tf.nn.relu(tf.matmul(G_h0, G_w1) + G_b1)

	G_w2 = tf.get_variable('G_w2', [G_h2_n, G_h3_n], initializer=w_init)
	G_b2 = tf.get_variable('G_b2', [G_h3_n], initializer=b_init)
	G_h2 = tf.nn.relu(tf.matmul(G_h1, G_w2) + G_b2)

	G_w3 = tf.get_variable('G_w3', [G_h3_n, G_output_n], initializer=w_init)
	G_b3 = tf.get_variable('G_b3', [G_output_n], initializer=b_init)
	G_output = tf.nn.sigmoid(tf.matmul(G_h2, G_w3) + G_b3)


	# G_h1 = tf.layers.dense(X, G_h1_n, activation=tf.nn.relu, 
	# 	kernel_initializer=w_init)
	# G_h2 = tf.layers.dense(G_h1, G_h2_n, activation=tf.nn.relu, 
	# 	kernel_initializer=w_init)
	# G_h3 = tf.layers.dense(G_h2, G_h3_n, activation=tf.nn.relu, 
	# 	kernel_initializer=w_init)
	# G_output = tf.layers.dense(G_h3, G_output_n, activation=tf.nn.sigmoid, 
	# 	kernel_initializer=w_init)

	return G_output

def discriminator(X):
	w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
	b_init = tf.constant_initializer(0.)

	D_w0 = tf.get_variable('D_w0', [D_input_n, D_h1_n], initializer=w_init)
	D_b0 = tf.get_variable('D_b0', [D_h1_n], initializer=b_init)
	D_h0 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, D_w0) + D_b0), 0.3)

	D_w1 = tf.get_variable('D_w1', [D_h1_n, D_h2_n], initializer=w_init)
	D_b1 = tf.get_variable('D_b1', [D_h2_n], initializer=b_init)
	D_h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(D_h0, D_w1) + D_b1), 0.3)

	D_w2 = tf.get_variable('D_w2', [D_h2_n, D_h3_n], initializer=w_init)
	D_b2 = tf.get_variable('D_b2', [D_h3_n], initializer=b_init)
	D_h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(D_h1, D_w2) + D_b2), 0.3)

	D_w3 = tf.get_variable('D_w3', [D_h3_n, D_output_n], initializer=w_init)
	D_b3 = tf.get_variable('D_b3', [D_output_n], initializer=b_init)
	D_output = tf.nn.sigmoid(tf.matmul(D_h2, D_w3) + D_b3)
	
	# D_h1 = tf.layers.dropout(tf.layers.dense(X, D_h1_n, activation=tf.nn.relu, 
	# 	kernel_initializer=w_init), 0.3)
	# D_h2 = tf.layers.dropout(tf.layers.dense(D_h1, D_h2_n, activation=tf.nn.relu, 
	# 	kernel_initializer=w_init), 0.3)
	# D_h3 = tf.layers.dropout(tf.layers.dense(D_h2, D_h3_n, activation=tf.nn.relu, 
	# 	kernel_initializer=w_init), 0.3)
	# D_output = tf.layers.dense(D_h3, D_output_n, activation=tf.nn.sigmoid, 
	# 	kernel_initializer=w_init)

	return D_output


with tf.variable_scope('G'):
	Z = tf.placeholder(tf.float32, shape=(None, G_input_n))
	G_z = generator(Z)

with tf.variable_scope('D') as scope:
	X = tf.placeholder(tf.float32, shape=(None, D_input_n))
	D_real = discriminator(X)
	scope.reuse_variables()
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

fixed_z = np.random.normal(0, 1, (8, 100))
with tf.Session() as sess:
	sess.run(init)

	print("Start of a training")
	start_tr = time.time()
	for epoch in range(epochs_n):
		G_losses = []
		D_losses = []
		start_epoch = time.time()
		
		imgs = sess.run(G_z, {Z: fixed_z})
		plt.figure(1, figsize=(20, 20))
		for j in range(len(imgs)):
			plt.subplot(2, 4, j + 1)
			plt.title("Img #" + str(j + 1))
			plt.imshow(np.reshape(imgs[j], [28, 28]), cmap='gray')
		plt.savefig("epoch-{}.png".format(epoch + 1))
		plt.close()

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
		print("Time spent for epoch #{}: {}".format(epoch + 1, end_epoch - start_epoch))

		# z = np.random.normal(0, 1, (8, 100))
		imgs = sess.run(G_z, {Z: fixed_z})
		plt.figure(1, figsize=(20, 20))
		for j in range(len(imgs)):
			plt.subplot(2, 4, j + 1)
			plt.title("Img #" + str(j + 1))
			plt.imshow(np.reshape(imgs[j], [28, 28]), cmap='gray')
		plt.savefig("epoch-{}.png".format(epoch + 1))
		plt.close()

		# if epoch % 10 == 0:
		save_path = saver.save(sess, "models/model" + str(epoch + 1) + ".ckpt")
		print("Model saved in path: %s" % save_path)

		print("")

	end_tr = time.time()
	print("Discriminator loss:", mean(D_losses))
	print("Generator loss:", mean(G_losses))
	print("Time spent for training:", end_tr - start_tr)