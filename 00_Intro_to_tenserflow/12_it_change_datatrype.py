import tensorflow as tf
import numpy as np

# 1. Create a tensor
B = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
C = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

print(B.dtype)
print(C.dtype)

# change to float16
B = tf.cast(B, dtype=tf.float16)
print(B.dtype)