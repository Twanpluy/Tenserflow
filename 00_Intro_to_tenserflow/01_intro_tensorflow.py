#import tensorflow as tf
import tensorflow as tf
# create tensor with tf.Variable

changeable_tensor = tf.Variable([10,7])
unchangeable_tensor = tf.constant([10,7])
print(changeable_tensor)
print(unchangeable_tensor)

# lets try to change one of the elements in our changeable tensor
changeable_tensor[0].assign(7)
print(changeable_tensor)
