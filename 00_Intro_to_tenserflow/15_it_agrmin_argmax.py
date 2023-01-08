import tensorflow as tf
import numpy as np

# create tensor
tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])

### positional maximum on which index , the maximum value is located
print(F)

# find argmax
print(F"agrmax : {tf.argmax(F)}")
#index of the maximum value
print (f"{F[tf.argmax(F)]}")
# check equality
print(F[tf.argmax(F)] == tf.reduce_max(F))


# min
print(F"agrmin : {tf.argmin(F)}")
#index of the minimum value
print (f"{F[tf.argmin(F)]}")
