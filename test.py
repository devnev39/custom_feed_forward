from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import model
import layers

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,784).astype('float32') / 255.0
x_test = x_test.reshape(-1,784).astype('float32') / 255.0

y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

mdl = model.model([
    layers.dense(256,activation='relu',input_shape=784),
    layers.dense(128,activation='relu'),
    layers.dense(10,activation='softmax')
],loss='mse')

loss = mdl.fit(x_train,y_train)
print(mdl.score(x_test,y_test))
plt.plot(loss)
plt.show()