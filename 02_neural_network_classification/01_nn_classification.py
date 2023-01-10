import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
from helperfunctions import plot_decision_boundary
### Create data to view and try to fit

# make circles
n_samples = 1000

# create circles
X, Y = make_circles(n_samples,noise=0.02, random_state=42)

# check  labels
# print(X)
# print(Y)

# plot data to check because we can't understand the data right now
circles = pd.DataFrame({"X0": X[:,0], "X1":X[:,1],"Label":Y})
print(circles.head())

# plot data
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.RdYlBu)
# plt.show()

# model purpose will become classify red osr blue dots.

# check data shapes
print(f" X:{X.shape}, Y: {Y.shape}")
# check length of data
print(f"length of data X: {len(X)}, length of data Y: {len(Y)}")
# view first example of data
print(f"first example X {X[0]}, first example Y: {Y[0]}")

### Steps in modelig
tf.random.set_seed(42)

model_red_blue_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),

])
# compile the model
model_red_blue_1.compile( 
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["accuracy"]
)
# fit the model
model_red_blue_1.fit(X,Y,epochs=200)

# evaluate the model
model_red_blue_1.evaluate(X,Y)

tf.random.set_seed(42)

model_red_blue_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1),
])
# compile the model
model_red_blue_2.compile( 
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["accuracy"]
)
# fit the model
model_red_blue_2.fit(X,Y,epochs=200)
# evaluate the model
model_red_blue_2.evaluate(X,Y)

#improve the model
tf.random.set_seed(42)

model_red_blue_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100,input_shape=(None,1)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
# compile the model
model_red_blue_3.compile(loss=tf.keras.losses.binary_crossentropy,
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
metrics=["accuracy"])

# fit the model
model_red_blue_3.fit(X,Y,epochs=200)

# evaluate the model
model_red_blue_3.evaluate(X,Y)

# plot the predictions
model_red_blue_3.predict(X)

plot_decision_boundary(model_red_blue_3,X,Y)

# problem with the models is that data is a circle, and our model is aline. 

# non linear data how to solve it?

