import tensorflow as tf
import numpy as np


# create tensor
B = tf.constant(np.random.randint(0, 100, size=50))
print(B)

# agrr tensor = condensing them from multiple values down to a smaller number of values
# Get absolutwe values:
print(tf.abs(B))

# Get the minimum value:
print(f"mix: {tf.reduce_min(B)}")

# Get the maximum value:
print(f"max: {tf.reduce_max(B)}")

# Get the mean value:
print(f"mean : {tf.reduce_mean(B)}")

# Get the sum value:
print(f"sum : {tf.reduce_sum(B)}")

print(f"shape: {B.shape}")
print(f"rank: {tf.size(B)}")
print(f"ndim: {B.ndim}")

# find variance 
print(f"variance: {tf.math.reduce_variance(tf.cast(B, dtype=tf.float32))}")