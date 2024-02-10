import pandas as pd
import numpy as np 

class LinearRegressor:

    def __init__(self, learning_rate=0.001, n_iter=1000) -> None:
        self.learning_rate = learning_rate
        self.n_iter=n_iter
        self.weigths=None
        self.bias=None
    
    def fit(self, X, y):
        num_samples, num_features =X.shape
        #Initialize weights with zero
        self.weigths=np.zeros(num_features)
        self.bias=0
        for _ in range(self.n_iter):
            y_pred=np.dot(X, self.weigths) + self.bias
            #calculating derivates weights and bias
            dW=(1/num_samples) * np.dot(X.T, (y_pred-y) )
            db=(1/num_samples) * np.sum(y_pred-y)
            #updating weigths and bias 
            try:
                self.weigths = self.weigths - (self.learning_rate * dW)
                self.bias = self.bias - (self.learning_rate * db)
            except RuntimeWarning as e:
                print("Caught a runtime warning:", e)

    def predict(self, X):
         y_pred=np.dot(X, self.weigths) + self.bias
         return y_pred



