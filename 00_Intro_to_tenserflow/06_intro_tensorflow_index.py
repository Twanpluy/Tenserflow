import tensorflow as tf
import numpy as np

#Get info for a tensor

# most imported info is the shape of the tensor
# shape  = tensor.shape
# Rank = tensor.ndim
# Axis or Dimension = tensor[0]
# Size = tf.size(tensor)

# Create a rank 4 tensor (4 dimensions)
rank_4_tensor = tf.zeros(shape=[2,3,4,5])

# Get various attributes of our tensor
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("total number of elements (2*3*4*5):", tf.size(rank_4_tensor).numpy())