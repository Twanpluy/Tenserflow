import tensorflow as tf
import numpy as np

# create tensor:
rank_4_tensor = tf.zeros(shape=[2,3,4,5])

# index tensor 

# Tensor can be index just like python list

# get first 2 elements
# rank_4_tensor[:2,:2,:2,:2]
# print("print first 2 elements:", rank_4_tensor[:2,:2,:2,:2])

# get the first element from each dimension from each index except for the final one
# rank_4_tensor[:1,:1,:1]
# print("print first 2 elements export final 1:", rank_4_tensor[:1,:1,:1])

# # added extra dimension to the end of our tensor (from rank 2)
rank_2_tensor = tf.constant([[10,7],[3,4]])

# print("rank_2_tensor:", rank_2_tensor)

# Get last item of each row of our rank 2 tensor
# rank_2_tensor[:-1, -1]
# print("rank_2_tensor[:-1, -1]:", rank_2_tensor[:, -1])

# add in extra dimension to rank 2 tensor
# ... every axis before the last one
rank_3_tensor = rank_2_tensor[..., tf.newaxis]
#from rank 2 to rank 3
print("rank_3_tensor:", rank_3_tensor)

#alt for tf.newaxis
tf.expand_dims(rank_2_tensor, axis=-1) # -1 means expand the final axis
# print("tf.expand_dims(rank_2_tensor, axis=-1):", tf.expand_dims(rank_2_tensor, axis=-1))

expant_0 = tf.expand_dims(rank_2_tensor, axis=0) # 0 means expand the first axis
print(expant_0)  


