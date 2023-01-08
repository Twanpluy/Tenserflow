import tensorflow as tf
import numpy as np


###Tensor flow intergration with numpy

# create a tensor directly from a numpy array
J = tf.constant(np.array([3., 7., 10.]))
print(J)

# convert a tensor to a numpy array
print(np.array(J), type(np.array(J)))