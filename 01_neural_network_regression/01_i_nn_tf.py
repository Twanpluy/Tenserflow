import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Start learing regression models
print(tf.__version__)
# Create data
X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
Y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

# Visualize data
plt.scatter(X,Y)
# plt.show()

# find relationship between X and Y
# check relationsship
print(X + 10 == Y)


## input and output shapes
house_info = tf.constant(["bedroom","bathroom","garage"])
house_price = tf.constant([939700])
print(house_info.shape, house_price.shape)



print(X[0], Y[0])
#trun np array into tensors
X = tf.constant(X,dtype=tf.float32)
Y = tf.constant(Y,dtype=tf.float32)


input_shape  = X[0].shape
output_shape = Y[0].shape

print(input_shape, output_shape)


# steps creating models in tensowflow
# 1. create a model - define the input and output layers, as well as the hidden layers of a deep learning model
# 2. compile model
# 3. fit model
# 4. evaluate model``
# set seed
tf.random.set_seed(42)

# create a model using the sequential API
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1)
# ])

# # compile the model
# model.compile(loss=tf.keras.losses.mae, #mae = mean abosule error
#               optimizer=tf.keras.optimizers.SGD(),
#               metrics=["mae"])


# # 3. Fit the model
# model.fit(tf.expand_dims(X,axis=1), Y, epochs=5)

# p = model.predict([17.0])
# print(p)

# improve model
# we can imporve our by altering steps
# 1. add more layers, increase the number of hidden units (neurons) within each of the hidden layers
# 2. change optizmizer, learning rate
# 3. fit leave training for  longer or more data
# model = tf.keras.Sequential([ 
#     tf.keras.layers.Dense(1)
# ])
# model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=["mae"])
# # (train longer)
# model.fit(tf.expand_dims(X,axis=1),Y,epochs=100)

# #
# p = model.predict([17.0])
# print(p)

# # change model with different optimizer
# model = tf.keras.Sequential([ 
#     tf.keras.layers.Dense(1)
# ])
# # change the optimizer to adam
# model.compile(loss=tf.keras.losses.mae,optimizer = tf.keras.optimizers.Adam(),metrics=["mae"])
# model.fit(tf.expand_dims(X,axis=1),Y,epochs=100)
# #
# p = model.predict([17.0])
# print(p)

# change model with adding layers
model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(50,activation=None),
    tf.keras.layers.Dense(1),

])
# change the optimizer to adam
model.compile(loss=tf.keras.losses.mae,optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),metrics=["mae"])
model.fit(tf.expand_dims(X,axis=1),Y,epochs=100)
model.summary()
#
p = model.predict([17.0])
print(p)