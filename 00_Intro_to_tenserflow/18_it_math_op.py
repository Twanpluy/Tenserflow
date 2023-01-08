import tensorflow as tf
import numpy as np


# create tensor
H = tf.range(1, 10)
print(H)

#  quearring , log, square root, and exponentiation
print(f"square: { tf.square(H)}")

print(f"sqrt: { tf.sqrt(tf.cast(H, dtype=tf.float32))}")

# find log
print(f"log: { tf.math.log(tf.cast(H, dtype=tf.float32))}")

# exponentiation
print(f"exp: { tf.exp(tf.cast(H, dtype=tf.float32))}")