import numpy as np
import copy
class layer:
    def __init__(self):
        self.prev = None
        self.connected =False
        self.output_shape = None
        self.input_shape = None
        self.units = None
        self.isInputLayer = False
        self.isOutputLayer = False

    def get_shape(self):
        if(self.connected):
            self.input_shape = [self.units,self.prev.input_shape[0]]
        else:
            if self.input_shape is None:
                raise Exception("Layer 0 not set")

            self.input_shape = [self.units,self.input_shape]

    def connect(self,prev):
        self.prev = prev
        self.connected = True
        self.get_shape()

class dense(layer):
    def __init__(self,units,activation='relu',input_shape=None,relu_suppress=0.01):
        super().__init__()
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.relu_suppress = relu_suppress
        self.dL = None
    
    def initialize(self):
        if self.input_shape is not None:
            self.W = np.random.standard_normal(self.input_shape)
            self.b = np.random.standard_normal([1,self.input_shape[0]])
        else:
            raise Exception("Shape not set")
    
    def process(self,inp):
        if self.isInputLayer:
            self.X = inp
        self.A = self.activate((np.dot(inp,self.W.T) + self.b))
        self.delta()
        return self.A
    
    def activate(self,z):
        if self.activation == 'sig':
            return np.nan_to_num(1/(1+np.exp(-z)))
        elif self.activation == 'tanh':
            return np.nan_to_num((2/(1+np.exp(-2*z))) - 1)
        elif self.activation == 'softmax':
            return np.nan_to_num(np.exp(z) / np.sum(np.exp(z),axis=1).reshape(-1,1))
        else:
            z[z<0] = 0.0
            return np.nan_to_num(z*self.relu_suppress)

    def delta(self):
        if self.activation == 'sig' or self.activation=='softmax':
            self.dZ = self.A*(1-self.A)
        elif self.activation == 'tanh':
            self.dZ = 1 - self.A**2
        else:
            a = copy.deepcopy(self.A)
            a[a>0] = self.relu_suppress
            self.dZ = a

    def loss_delta(self,*args):
        if(len(args)==2):
            if(self.isOutputLayer):
                if(args[0]=='mse'):
                    self.dL = -2*(args[1]-self.A)/len(args[1])
                elif(args[0] == 'CCE'):
                    self.dL = (self.A - args[1])/len(args[1])
                    
                    
        elif(len(args)==3):
            self.dL = np.dot(args[0] * args[1],args[2])

    def apply_grads(self,lr):
        if self.isInputLayer:
            self.W -= np.dot((self.dL * self.dZ).T,self.X) * lr
            self.b -= np.mean(self.dL * self.dZ,axis=0,keepdims=True) * lr
        else:
            self.W -= np.dot((self.dL * self.dZ).T,self.prev.A) * lr
            self.b -= np.mean(self.dL * self.dZ,axis=0,keepdims=True) * lr
        