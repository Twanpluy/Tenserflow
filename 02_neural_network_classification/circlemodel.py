import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
from helperfunctions import plot_decision_boundary
# make circles
n_samples = 1000
# need to create data
X, Y = make_circles(n_samples,noise=0.02, random_state=42)
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(100,activation="ReLU"),
    tf.keras.layers.Dense(100,activation="ReLU"),

])

#COMPILE THE MODEL
model_4.compile(loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    metrics=["accuracy"])

history = model_4.fit(X,Y,epochs=100)

plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.RdYlBu)
plt.show()

plot_decision_boundary(model_4,X,Y)