import tensorflow as tf
import numpy as np

# Basic Operations
# +, -, *, /
# create tensor:
rank_2_tensor = tf.constant([[10,7],[3,4]])
# +
print(rank_2_tensor + 10)
# *
print(rank_2_tensor * 10)
# -
print(rank_2_tensor - 10)
# /
print(rank_2_tensor / 10)


# we can use tensorflow built-in functions
# by using function it is faster than using python operator
tf.multiply(rank_2_tensor, 10)

tf.add(rank_2_tensor, 10)

tf.subtract(rank_2_tensor, 10)

tf.divide(rank_2_tensor, 10)
