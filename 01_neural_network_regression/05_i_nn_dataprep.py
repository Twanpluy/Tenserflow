import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# import data 
# https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
DATA_LINK = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
insurance = pd.read_csv(DATA_LINK)

print(insurance.head())
#normalise data with skikit learn

# create column transformer
ct = make_column_transformer( (
    MinMaxScaler(),["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
    )

X = insurance.drop("charges", axis=1)
Y = insurance["charges"]

# train and test splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# fit the column transformer to our training data
ct.fit(X_train)

# transform training and trest data with normalization (MinMaxScaler) and OneHotEncoder
X_train_norm = ct.transform(X_train)
X_test_norm = ct.transform(X_test)

# create new models
tf.random.set_seed(42)
# 1. create a model
insurance_model = tf.keras.Sequential([
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(1)
                ])

# 2. compile the model with adam lr 0.01
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(lr=0.01),
                        metrics=["mae"])

# 3. fit the model
insurance_model.fit(X_train_norm, Y_train, epochs=100)

# 4. evaluate the model
insurance_model.evaluate(X_test_norm, Y_test)

