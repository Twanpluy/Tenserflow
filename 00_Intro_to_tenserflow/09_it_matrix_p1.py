import tensorflow as tf
import numpy as np

# Matrix Multiplication
# matrix multiplication is most common operation in ML
# create tensor:
# rank_2_tensor = tf.constant([[10,7],[3,4]])

# # matrix multiplication
# print(tf.matmul(rank_2_tensor, rank_2_tensor))

# # matrix multiplication with python operator
# print(rank_2_tensor @ rank_2_tensor)

# # create 3 by 3 tensor  
# rank_3_tensor = tf.constant([[[1,2,5],[7,2,1],[3,3,3]]])
# # results:
# # print(rank_3_tensor)

# # create new rank_2_tensor [3,5],[6,7],[1,8]
# tensor = tf.constant([[3,5],[6,7],[1,8]])

# # multiply matrix
# print(tf.matmul(rank_3_tensor, tensor))

# tensor multiplication withb diffent shapes
# create tensor
X = tf.constant([[1,2],[3,4],[5,6]])
Y = tf.constant([[7,8],[9,10],[11,12]])

# multiply
# print(tf.matmul(X,Y))
