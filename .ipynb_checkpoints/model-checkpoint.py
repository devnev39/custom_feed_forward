import numpy as np
import math
def vectorize(y):
    out = np.zeros((len(y),10))
    for i,each in enumerate(y):
        out[i][each] = 1
    
    return out

class model:
    def __init__(self,lst : list,loss='mse'):
        self.lst = lst
        self.loss = loss
        self.prepare()
    
    def prepare(self):
        self.lst[0].isInputLayer = True
        self.lst[-1].isOutputLayer = True
        self.lst[0].get_shape()
        self.lst[0].initialize()
        for i in range(1,len(self.lst)):
            self.lst[i].connect(self.lst[i-1])
            self.lst[i].initialize()
    
    def set_y(self):
        if np.max(self.y) == 1:
            if len(self.y.shape) == 1:
                self.y = self.y[...,np.newaxis]
            else:
                return
        else:
            self.y_excl = self.y
            self.y = vectorize(self.y)
            self.lst[-1].true_preds = self.y_excl
            self.lst[-1].true_preds_onehot = self.y
             
    
    def fit(self,X,y,epochs=10,lr=0.01,batch_size = 64):
        self.lr = lr
        loss = []
        for each in range(epochs):        
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
            print(f'Epoch : {each} , loss : {loss[-1]}')
        return loss
    
    def forward_pass(self):
        a = None
        for layer in self.lst:
            if a is None:
                a = layer.process(self.X)
                continue
            a = layer.process(a)
        
        self.__current_output = a

    def backward_pass(self):
        for i in range(len(self.lst)-1,-1,-1):
            if i == len(self.lst)-1:
                self.lst[i].loss_delta(self.loss,self.y)
                continue
            self.lst[i].loss_delta(self.lst[i+1].dL,self.lst[i+1].dZ,self.lst[i+1].W)

    def gradient_descent(self):
        self.forward_pass()
        self.backward_pass()

        for layer in self.lst:
            layer.apply_grads(self.lr)
    
    def __raw_forward_pass(self,X):
        a = None
        for e in self.lst:
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
            return (self.lst[-1].A > 0.5).astype('int')
        else:
            ans = np.zeros((len(X),self.lst[-1].units))
            for i,e in enumerate(self.lst[-1].A):
                ans[i][np.argmax(e)] = 1
            return ans
                
    def Loss(self):
        if self.loss=='mse':
            return np.mean((self.y-self.__current_output)**2)
        elif self.loss == 'SCCE':
            return np.mean(-np.log(np.take(self.__current_output,self.y_excl)))
        elif self.loss == 'CCE':
            return np.mean(-np.sum(self.y*np.log(self.__current_output),axis=1))
    
    def score(self,X,y):
        y_pred = self.predict(X)
        if np.max(y) > 1:
            y = vectorize(y)
        lst = []
        for e,y in zip(y_pred,y):
            lst.append(np.array_equal(e,y))
        return np.mean(lst)
                