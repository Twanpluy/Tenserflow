import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd
from helperfunctions import plot_decision_boundary
from sklearn.model_selection import train_test_split
# make circles
n_samples = 1000
# need to create data
X, Y = make_circles(n_samples,noise=0.02, random_state=42)
model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation="ReLU"),
    tf.keras.layers.Dense(4,activation="ReLU"),
    tf.keras.layers.Dense(1,),
])

#COMPILE THE MODEL
model_6.compile(loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    metrics=["accuracy"])

history = model_6.fit(X,Y,epochs=250)

model_6.evaluate(X,Y)

plot_decision_boundary(model_6,X,Y)


model_7 = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation="ReLU"),
    tf.keras.layers.Dense(4,activation="ReLU"),
    tf.keras.layers.Dense(1,activation="sigmoid"),
])

model_7.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=0.01),
            metrics=["accuracy"])

history = model_7.fit(X,Y,epochs=2500)

model_7.evaluate(X,Y)

plot_decision_boundary(model_7,X,Y)