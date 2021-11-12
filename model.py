import numpy as np
import math
from sklearn.utils import shuffle
def vectorize(y):
    out = np.zeros((len(y),10))
    for i,each in enumerate(y):
        out[i][each] = 1
    
    return out

class model:
    def __init__(self,lst : list,loss='mse'):
        self.__lst = lst
        self.loss = loss
        self.prepare()
    
    def prepare(self):
        self.__lst[0].isInputLayer = True
        self.__lst[-1].isOutputLayer = True
        self.__lst[0].get_shape()
        self.__lst[0].initialize()
        for i in range(1,len(self.__lst)):
            self.__lst[i].connect(self.__lst[i-1])
            self.__lst[i].initialize()
    
    def set_y(self):
        if np.max(self.y) == 1:
            if len(self.y.shape) == 1:
                self.y = self.y[...,np.newaxis]
            else:
                return
        else:
            self.y_excl = self.y
            self.y = vectorize(self.y)
            #self.__lst[-1].y_excl = self.y_excl
             
    
    def fit(self,X,y,epochs=10,lr=0.01,batch_size=64,val_data=None):
        self.lr = lr
        loss = []
        for each in range(epochs):   
            shuffle(X,y,random_state=0)
            start = 0
            completed = len(X)
            batch_loss = []
            while(completed):
                end = start + min(batch_size,completed)
                self.X = X[start:end]
                self.y = y[start:end]
                #print(completed)
                self.set_y()
                
                self.gradient_descent()
                batch_loss.append(self.Loss())
                  
                completed -= (end-start)
                start = end
                if completed <= 0:
                    completed = 0
            loss.append(np.mean(batch_loss))   
            if val_data != None:
                sc = self.score(val_data[0],val_data[1])
                print(f'Epoch : {each} , loss : {loss[-1]} , val_acc : {sc}')
            else:
                print(f'Epoch : {each} , loss : {loss[-1]}')
        return loss
    
    def forward_pass(self):
        a = None
        for layer in self.__lst:
            if a is None:
                a = layer.process(self.X)
                continue
            a = layer.process(a)
        
        self.__current_output = a

    def backward_pass(self):
        for i in range(len(self.__lst)-1,-1,-1):
            if i == len(self.__lst)-1:
                self.__lst[i].loss_delta(self.loss,self.y)
                continue
            self.__lst[i].loss_delta(self.__lst[i+1].dL,self.__lst[i+1].dZ,self.__lst[i+1].W)

    def gradient_descent(self):
        self.forward_pass()
        self.backward_pass()

        for layer in self.__lst:
            layer.apply_grads(self.lr)
    
    def __raw_forward_pass(self,X):
        a = None
        for e in self.__lst:
            if a is None:
                a = e.process(X)
                continue
            a = e.process(a)
        return a
    
    def raw_predict(self,X):
        return self.__raw_forward_pass(X)

    def predict(self,X):
        self.X = X
        self.forward_pass()
        if np.max(self.y_excl) == 1:
            return (self.__lst[-1].A > 0.5).astype('int')
        else:
            ans = np.zeros((len(X),self.__lst[-1].units))
            for i,e in enumerate(self.__lst[-1].A):
                ans[i][np.argmax(e)] = 1
            return ans
                
    def Loss(self):
        if self.loss=='mse':
            return np.mean((self.y-self.__current_output)**2)
        elif self.loss == 'CCE':
            return np.mean(-np.sum(self.y*np.log(self.__current_output+10e-5),axis=1))
    
    def score(self,X,y):
        y_pred = self.predict(X)
        if np.max(y) > 1:
            y = vectorize(y)
        lst = []
        for e,y in zip(y_pred,y):
            lst.append(np.array_equal(e,y))
        return np.mean(lst)
    
    def get_lst(self):
        return self.__lst
                