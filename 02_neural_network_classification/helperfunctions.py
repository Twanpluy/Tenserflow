# helper functions to visualize the data and the decision boundaries of the models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

X = np.arange(-100,100,0.1)
Y = np.arange(-100,100,0.1)

def plot_decision_boundary(model,X,Y):
    """
    Plots the decision boundary created by a model predicting on X.
    """
    # define x_min and x_max
    x_min = X[:,0].min() - 0.1
    x_max = X[:,0].max() + 0.1

    # define y_min and y_max
    y_min = X[:,1].min() - 0.1
    y_max = X[:,1].max() + 0.1

    ## create a meshgrid
    xx,yy = np.meshgrid(np.linspace(x_min,x_max,100),
                        np.linspace(y_min,y_max,100))

    # create X values 
    x_in = np.c_[xx.ravel(),yy.ravel()] # stack 2D arrays together

    # 
    y_pred = model.predict(x_in)
    
    # make predicitions
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        y_pred = np.argmax(y_pred,axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    #plot the decision boundary
    plt.contourf(xx,yy,y_pred,cmap=plt.cm.RdYlBu,alpha=0.7)
    plt.scatter(X[:,0],X[:,1],c=Y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.show()
