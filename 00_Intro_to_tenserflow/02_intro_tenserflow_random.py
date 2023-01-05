#import tensorflow as tf
import tensorflow as tf

# create two random (but the same) tensors
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))
print(random_1)

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))

random_3 = tf.random.Generator.from_seed(42)
random_3 = random_3.uniform(shape=(3,2))

# are they equal?
print(random_1 == random_2)
print(random_1 == random_3)