import tensorflow as tf
import numpy as np

# matrix multi is also called dot product
# tf.matmul(matrix1, matrix2) #
# tf.tensordot(matrix1, matrix2, axes=1) # axes=1 is the same as matrix multi 

# 1. Create a matrix
X = tf.constant([[1,2],[3,4],[5,6]])
Y = tf.constant([[7,8],[9,10],[11,12]])

tensordot = tf.tensordot(tf.transpose(X),Y, axes=1)
print(tensordot)
tensordot = tf.tensordot(X,tf.transpose(Y), axes=1)
print(tensordot)

# with matmul
matmul = tf.matmul(tf.reshape(X,shape=(2,3)),Y)
print(matmul)
matmul = tf.matmul(X,tf.reshape(Y,shape=(2,3)))
print(matmul)

# change values Y ,
print(f"Normal Y: {Y}" )
print(f"Respape Y: {tf.reshape(Y,shape=(2,3))}")
print(f"Transpose Y: {tf.transpose(Y)}")