import tensorflow as tf
import numpy as np

# matrix rules:
# 1. inner dimensions must match
# 2. result has shape of outer dimensions

# matrixs:
X = tf.constant([[1,2],[3,4],[5,6]])
Y = tf.constant([[7,8],[9,10],[11,12]])

# change shape matrix Y
# Y = tf.reshape(Y, [2,3]) # 2 rows, 3 columns
# X = tf.reshape(X, [2,3]) # 3 rows, 2 columns

# multiply
# print(tf.matmul(tf.reshape(X,[2,3]),Y))

# print(tf.reshape(X,[2,3]))
# # Transpoose
# print(tf.transpose(X))

# transpose matrix
print(tf.matmul(tf.transpose(X),Y))