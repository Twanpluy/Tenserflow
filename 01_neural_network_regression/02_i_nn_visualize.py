import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Start learing regression models
print(tf.__version__)
# Create data 
X = tf.range(-100,100,4)

Y = X + 15

# vizuallize the model
plt.plot(X,Y)
# plt.show()

# # 3 sets but for tut only 2
# X_train = X[:40]
# # test
# X_test = X[40:]
# print(X_train.shape, X_test.shape)

# check length data
len(X)

# split data into train and test sets
X_train = X[:40]
X_test = X[40:]

Y_train = Y[:40]
Y_test = Y[40:]

# print(len(X_train), len(X_test), len(Y_train), len(Y_test))

# # Visualize the data in 1 plot
# plt.figure(figsize=(10,7))
# # blue
# plt.scatter(X_train, Y_train, c="b", label="Training data")
# # green
# plt.scatter(X_test, Y_test, c="g", label="Testing data")
# plt.legend()
# # plt.show()

# # set seed
# tf.random.set_seed(42)
# create a model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(50,input_shape=[1] , name="input_layer"),

    tf.keras.layers.Dense(1,name="output_layer")
    
], name="MODEL_1")

model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                metrics=["mae"])
# model_1.build(input_shape=(None,1))

# total params :L total number of parameters in the model
# Trainable params: total number of parameters in the model that can be updated while training
# Non-trainable params: total number of parameters in the model that cannot be updated while training

model_1.fit(X_train, Y_train, epochs=1000,verbose=0)

# p = model_1.predict([17.0])
# print(p)
# model_1.summary()
# # import plotmodel
# from keras.utils.vis_utils import plot_model
# tf.keras.utils.plot_model(
# model_1, to_file='model.png', show_shapes=True, show_dtype=False,
# show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
# )
# to visualize the model good idea to plot them vs groud truth vs model prediction
y_pred = model_1.predict(X_test)
print(y_pred)

# create plotting function
def plot_predictions(train_data = X_train,
                     tain_label = Y_train,
                     test_data = X_test,
                     test_label = Y_test,                
                     predictions = y_pred):
    """[plot training data, test data and compare predictions]"""
    plt.figure(figsize=(10,7))
    #plot training data in blue
    plt.scatter(train_data, tain_label, c="b", label="Training data")
    #plot test data in green
    plt.scatter(test_data, test_label, c="g", label="Testing data")
    #plot model predictions in red
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()  

# plot_predictions()    

## evalute the model
print(model_1.evaluate(X_test, Y_test))

# cal mean absolute error
mae = tf.metrics.mean_absolute_error(y_true=Y_test,y_pred=tf.constant(tf.squeeze(y_pred)))

# calcualte mean square error
mse = tf.metrics.mean_squared_error(y_true=Y_test,y_pred=tf.constant(tf.squeeze(y_pred)))
print(mse)

def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true,y_pred=tf.squeeze(y_pred))

def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true,y_pred=tf.squeeze(y_pred))