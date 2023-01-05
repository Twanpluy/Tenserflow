# in this noteboob, we are going to cover basic tensor
# import tensorflow
import tensorflow as tf

#print version
tf.__version__

# create a constant
scalar = tf.constant(7)
scalar

# check the number of dimension of aa  tensor
scalar.ndim

#create a vector
vector = tf.constant([10,10])
vector

#check the dimension of vector
vector.ndim

#create a matrix
matrix = tf.constant([[10,7],[7,10]])
matrix
matrix.ndim

# craete another matrix with other type
another_matrix = tf.constant([ [10,7],[3.,2.],[8.,9.]], dtype=tf.float16)
another_matrix
another_matrix.ndim

# create a tensor
tensor = tf.constant([[[1,2,3],[4,5,6]], 
                        [[7,8,9],[10,11,12]],
                        [[13,14,15],[16,17,18]]])
print(tensor)
print(tensor.ndim)

# what we created
# scalar: a single number
# vector: a number with direction
# matrix: a 2-dimensional array of numbers
# tensor: an n-dimensional array of numbers (when n can be any number, a 0-d tensor is a scalar, a 1-d tensor is a vector)
