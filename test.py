import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from tensorflow.keras.datasets import mnist

import model
import layers

# X,y = make_blobs(centers=2,random_state=42)
# y = y.reshape(-1,1)

def vectorize(y):
    out = np.zeros((len(y),10))
    for i,each in enumerate(y):
        out[i][each] = 1
    return out

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,784).astype('float32') / 255.0
x_test = x_test.reshape(-1,784).astype('float32') / 255.0
#print(y_train[2])
y_train = vectorize(y_train)
y_test  = vectorize(y_test)

# print(y_train.shape)
# print(y_train[2])
# sys.exit()

mdl = model.model([
    layers.dense(128,activation='sig',input_shape=784),
    layers.dense(64,activation='sig'),
    layers.dense(10,activation='sig')
])

loss = mdl.fit(x_train,y_train,epochs=5)
#print(mdl.predict(x_test[0]))
print(mdl.raw_predict().shape)
print(y_train.shape)
mdl.forward_pass()

print(x_train[0])

plt.plot(loss)
plt.show()
print(mdl.score(x_test,y_test))