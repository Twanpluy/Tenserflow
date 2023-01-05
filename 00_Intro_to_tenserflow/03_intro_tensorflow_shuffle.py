#import tensorflow as tf
import tensorflow as tf

# how will we shuffle our tensor? (for when you want to shuffle your data so the inherent order doesn't affect learning)
not_shuffled = tf.constant([[10,7],[3,4],[2,5]])
not_shuffled.ndim

tf.random.set_seed(42) # global level random seed
shuffled = tf.random.shuffle(not_shuffled,seed=42)
print(shuffled)


# shuffle a tensor (valuable for when you want to shuffle your data so the inherent order doesn't affect learning)
# If neither the global seed nor the operation seed is set: A randomly picked seed is used for this op.
# If the global seed is set, but the operation seed is not: The system deterministically picks an operation seed in conjunction with the global seed so that it gets a unique random sequence. Within the same version of tensorflow and user code, this sequence is deterministic. However across different versions, this sequence might change. If the code depends on particular seeds to work, specify both global and operation-level seeds explicitly.
# If the operation seed is set, but the global seed is not set: A default global seed and the specified operation seed are used to determine the random sequence.
# If both the global and the operation seed are set: Both seeds are used in conjunction to determine the random sequence.

# create 5 random tensors of the same shape (but they can be any shape)
# random 1
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))
# random 2
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))
# random 3
random_3 = tf.random.Generator.from_seed(42)
random_3 = random_3.uniform(shape=(3,2))
# random 4
random_4 = tf.random.Generator.from_seed(42)
random_4 = random_4.normal(shape=(3,2))
# random 5
random_5 = tf.random.Generator.from_seed(42)
random_5 = random_5.uniform(shape=(3,2))


#shuffle tensors in a  same way
tf.random.set_seed(1)
random_1_shuffled = tf.random.shuffle(random_1)
random_2_shuffled = tf.random.shuffle(random_2,seed=2)

print(random_1_shuffled)
print(random_2_shuffled)
