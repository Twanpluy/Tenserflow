import tensorflow as tf
import numpy as np

#list of indices
some_list = [0, 1, 2, 3] 

#one hot encoding
H = tf.one_hot(some_list, depth=4)
print(H)

#spcify custom values for on and off values
tf.one_hot(some_list,depth=4,on_value="dance",off_value="not dance")
