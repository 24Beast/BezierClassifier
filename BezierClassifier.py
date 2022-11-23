import numpy as np


class BezierClassifier:

    def __init__(self,max_points=5):
        self.max_points = max_points
        self.epsilon = (0.1)**10
        self.weights = np.arange(0,1+self.epsilon,(1/(max_points-1)))
    
    def preprocess(self,X):
        min_x = np.min(X,axis=1)
        max_x = np.max(X,axis=1)
        max_x[min_x==max_x] += self.epsilon 
        return (X-min_x)/(max_x-min_x)
    
    def combin(n,i):
        return np.math.factorial(n)/(np.math.factorial(n-i)*np.math.factorial(i))

    def initWeights(self):
        l = self.max_points-1
        self.w = np.zeros((l+1,l+1))
        for k in range(l+1):
            for i in range(k+1):
                self.w[k,i] = self.combin(l,i) * ((-1)**(k-i)) * self.combin(l-i,l-k)
        return self