import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""[plot training data, test data and compare predictions]"""
def plot_predictions(train_data ,
                     tain_label ,
                     test_data ,
                     test_label ,                
                     predictions ):
    plt.figure(figsize=(10,7))
    #plot training data in blue
    plt.scatter(train_data, tain_label, c="b", label="Training data")
    #plot test data in green
    plt.scatter(test_data, test_label, c="g", label="Testing data")
    #plot model predictions in red
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()  

"""[Calculate perfomance metrics for regression models using MAE]"""
def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true,y_pred=tf.squeeze(y_pred))
"""[Calculate perfomance metrics for regression models using MSE]"""
def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true,y_pred=tf.squeeze(y_pred))
