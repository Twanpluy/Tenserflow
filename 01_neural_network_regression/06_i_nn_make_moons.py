import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import seaborn as sn

make_moons_set_1 = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=42)

# create X and Y
X, Y = make_moons_set_1

# split data
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# create a model
tf.random.set_seed(42)
make_moons_model = tf.keras.Sequential([
                tf.keras.layers.Dense(300),
                tf.keras.layers.Dense(200),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(1)
                ])
# compile the model
make_moons_model.compile(loss=tf.keras.losses.binary_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["accuracy"])
# fit the model
# history = make_moons_model.fit(X_train, Y_train, epochs=150)
history = make_moons_model.fit(X_train, Y_train, epochs=150)

# evaluate the model
make_moons_model.evaluate(X_test, Y_test)
# plot the loss curve
pd.DataFrame(history.history).plot()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss curve")
plt.show()

