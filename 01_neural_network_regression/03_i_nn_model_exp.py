import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from hf import plot_predictions, mae, mse
import pandas as pd
# Create data 
X = tf.range(-100,100,4)
Y = X + 15

# split data into train and test sets
X_train = X[:40]
X_test = X[40:]

Y_train = Y[:40]
Y_test = Y[40:]

# set random seed
tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1,input_shape=[1])
])
model_1.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["mae"]
)

# model_1.summary()
model_1.fit(X_train, Y_train, epochs=100)

# model  2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=[1]),
    tf.keras.layers.Dense(1,input_shape=[1]),
])
model_2.compile(    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["mae"])
model_2.fit(X_train, Y_train, epochs=100)

# model  3
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=[1]),
    tf.keras.layers.Dense(1,input_shape=[1]),
])
model_3.compile(    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=["mae"])
model_3.fit(X_train, Y_train, epochs=500)

# # MAKE PREDICTIONS AND PLT MODEL1
Y_pred_1 = model_1.predict(X_test)
plot_predictions(train_data = X_train,
                 tain_label = Y_train,
                 test_data =  X_test,
                 test_label = Y_test,                
                 predictions =Y_pred_1)
mae_1 = mae(Y_test, Y_pred_1)
msi_1 = mse(Y_test, Y_pred_1)

print(f"MAE: {mae_1}")
print(f"MSE: {msi_1}")

print(f"model2 summary: {model_2.summary()}")
# # MAKE PREDICTIONS AND PLT MODEL2
Y_pred_2 = model_2.predict(X_test)

plot_predictions(train_data = X_train,
                 tain_label = Y_train,
                 test_data =  X_test,
                 test_label = Y_test,                
                 predictions =Y_pred_2)
mae_2 = mae(Y_test, Y_pred_2)
msi_2 = mse(Y_test, Y_pred_2)

print(f"MAE 2: {mae_2}")
print(f"MSE 2: {msi_2}")

# # MAKE PREDICTIONS AND PLT MODEL3
Y_pred_3 = model_3.predict(X_test)

plot_predictions(train_data = X_train,
                 tain_label = Y_train,
                 test_data =  X_test,
                 test_label = Y_test,                
                 predictions =Y_pred_3)
mae_3 = mae(Y_test, Y_pred_3)
msi_3 = mse(Y_test, Y_pred_3)

print(f"MAE 3: {mae_3}")
print(f"MSE 3: {msi_3}")

# create compare models
model_resuls  = [ ["model_1",mae_1.numpy() ,msi_1.numpy()],
                  ["model_2",mae_2.numpy(),msi_2.numpy()],  
                  ["model_3",mae_3.numpy(),msi_3.numpy()],]

all_results = pd.DataFrame(model_resuls, columns=["model","mae","mse"])

print(all_results)
#model 2 works best

# track prediciotns with tools
# **Tensorboard** and **tf.keras.callbacks**.
# Weight and bias tracking with TensorBoard

# save models to use outside of python
# model_2.save("model_2")
# model_2.save("h5_model_2.h5")


#load models 
load_model_2 = tf.keras.models.load_model("h5_model_2.h5")
print(load_model_2.summary())