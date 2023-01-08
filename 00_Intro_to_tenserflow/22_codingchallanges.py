import tensorflow as tf
import numpy as np

# print tf version 
print(f"tf version: {tf.__version__}")

#load dataset imigas keras 
mnist = tf.keras.datasets.mnist
#train model
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#model 
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

print(model.summary())

# predict
predictions = model(x_train[:1]).numpy()
print(predictions)

#logits:
print(tf.nn.softmax(predictions).numpy())

#loss function 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn) # <tensorflow.python.keras.losses.SparseCategoricalCrossentropy object at 0x0000020B1B2B0D30>
# normal loss function readable
print(loss_fn(y_train[:1], predictions).numpy())

#compile model
model.compile(optimizer="adam",loss=loss_fn,metrics=["accuracy"])
model.fit(x_train,y_train,epochs=5)
model.evaluate(x_test,y_test,verbose=2)

# probability model
probability_model = tf.keras.Sequential(
    [
        model,tf.keras.layers.Softmax()
    ]
)
# print probability model
print(probability_model(x_test[:5]))