import tensorflow as tf
import numpy as np

# create a tensor using numpy
numpy_t_one = tf.ones([10,7])
numpy_t_zero = tf.zeros([10,7])

print(f"tf ones {numpy_t_one}")
print(f"tf zeroes : {numpy_t_zero}")

# numpy array to tensor
numpy_array = np.arange(1,25,dtype=np.int32)
print(f"numpy array : {numpy_array}")

A = tf.constant(numpy_array,shape=(2,3,4))
print(f"tensor from numpy array : {A}")

