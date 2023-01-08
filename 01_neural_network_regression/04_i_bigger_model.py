import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from hf import plot_predictions, mae, mse
import pandas as pd
from sklearn.model_selection import train_test_split

# import data 
# https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
DATA_LINK = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
insurance = pd.read_csv(DATA_LINK)

# SET SEX DATA TO INTEGER with one_hot_encoding
insurance_one_hot = pd.get_dummies(insurance)

# create X and Y
# drop CHARGES from X TO CREATE Y
X = insurance_one_hot.drop("charges", axis=1)
Y = insurance_one_hot["charges"]

# split data into train and test sets
# SPLIT IN 80% TRAIN AND 20% TEST 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))


# create a model wit
#set random seed
tf.random.set_seed(42)
insurance_model = tf.keras.Sequential([
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(1)
                ])

#compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                        metrics=["mae"])
# fit the model
insurance_model.fit(X_train, Y_train, epochs=100)                        


# add extra expiriments
insurance_model_2 = tf.keras.Sequential([
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(1)
                ])

# compile the model_2
insurance_model_2.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])
# fit the model_2
insurance_model_2.fit(X_train, Y_train, epochs=100)

# add more expiriments model 3
insurance_model_3 = tf.keras.Sequential([
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(1)
                ])

# compile the model_3
insurance_model_3.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])

# fit the model_3
insurance_model_3.fit(X_train, Y_train, epochs=250)   

# add more expiriments model 4
insurance_model_4 = tf.keras.Sequential([
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(1)
                ])
# compile the model_4 change to adam    optimizer=tf.keras.optimizers.Adam()
insurance_model_4.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])
# fit the model_4
history = insurance_model_4.fit(X_train, Y_train, epochs=500)   


# evaluate the model
e_im = insurance_model.evaluate(X_test, Y_test)
e_im_2 = insurance_model_2.evaluate(X_test, Y_test)
e_im_3 = insurance_model_3.evaluate(X_test, Y_test)
e_im_4 = insurance_model_4.evaluate(X_test, Y_test)

print(f"model_1: {e_im} model_2: {e_im_2} model_3: {e_im_3} model_4: {e_im_4}")  

#  weird nan in my models 
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()