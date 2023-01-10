import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
from helperfunctions import plot_decision_boundary
### Create data to view and try to fit

tf.random.set_seed(42)
X_regression = tf.range(0,1000,5)
Y_regression = tf.range(100,1100,5) # Y = X + 100

# Train and test split
X_train = X_regression[:150]
Y_train = Y_regression[:150]

X_test = X_regression[150:]
Y_test = Y_regression[150:]

# model_red_blue_4 = tf.keras.Sequential([
#     tf.keras.layers.Dense(10, input_shape=(1,)),
#     tf.keras.layers.Dense(1)
# ])

# # compile the model
# model_red_blue_4.compile(loss=tf.keras.losses.mae,
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=["mae"])

# model_red_blue_4.fit(X_train,Y_train,epochs=100)
# # evaluate the model
# model_red_blue_4.evaluate(X_test,Y_test)
# # plot_decision_boundary(model_red_blue_3,X_regression,Y_regression)

# # plot the predictions
# y_reg_pred = model_red_blue_4.predict(X_test)
# plt.scatter(X_test,Y_test)
# plt.scatter(X_test,y_reg_pred)
# plt.show()

