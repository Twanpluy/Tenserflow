import tensorflow as tf
import numpy as np

G = tf.constant(tf.random.uniform(shape=[50]), shape=(1, 1, 1, 1, 50))

print(G)

#sqeeze
G_squeezed = tf.squeeze(G)
print(G_squeezed)

# sqeeze
print(G_squeezed.shape)
print(G.shape)