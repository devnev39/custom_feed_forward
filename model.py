import numpy as np

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
    
    def fit(self,X,y,epochs=10,lr=0.01):
        self.X = X
        self.y = y
        self.lr = lr
        loss = []
        for each in range(epochs):
            self.gradient_descent()
            loss.append(self.Loss())
            print(f'Epoch : {each} , loss : {loss[-1]}')
        
        return loss
    
    def forward_pass(self):
        a = None
        for layer in self.lst:
            if a is None:
                a = layer.process(self.X)
                continue
            a = layer.process(a)

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
        
    def __predict(self):
        self.forward_pass()
        return self.lst[-1].A
    
    def raw_predict(self):
        return self.__predict()

    def predict(self,X):
        self.X = X
        self.forward_pass()
        return (self.lst[-1].A > 0.5).astype('int')

    def Loss(self):
        return np.mean((self.y-self.__predict())**2)
    
    def score(self,X,y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)
        



        