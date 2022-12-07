import numpy as np
import nevergrad as ng
from sklearn.metrics import log_loss

class BezierClassifier:

    def __init__(self,max_points=5,num_epochs=1000,num_workers=1):
        self.max_points = max_points
        self.epsilon = (0.1)**10
        self.num_instances = None
        self.num_dimensions = None
        self.buget = num_epochs
        self.n_workers = num_workers
        self.instrumentation = None
        self.optimizer = None
    
    def preprocess(self,X):
        min_x = np.min(X,axis=1)
        max_x = np.max(X,axis=1)
        max_x[min_x==max_x] += self.epsilon 
        return (X-min_x)/(max_x-min_x)
    
    
    def combin(self,n,i):
        return np.math.factorial(n)/(np.math.factorial(n-i)*np.math.factorial(i))


    def initWeights(self):
        l = self.max_points-1
        self.w = np.zeros((l+1,l+1))
        for k in range(l+1):
            for i in range(k+1):
                self.w[i,k] = self.combin(l,i) * ((-1)**(k-i)) * self.combin(l-i,l-k)
        if(self.num_instances==None or self.num_dimensions==None):
            print("Please Set num_instances and num_dimensions parameters before weight initialization")
            return self
        self.t = np.random.random((self.num_instances,self.num_dimensions))
        self.t = np.cumsum(self.t/np.sum(self.t,axis=0),axis=0)
        self.p = np.random.random(((self.num_dimensions,self.max_points))).T
        self.p = np.cumsum(self.p/np.sum(self.p,axis=0),axis=0).T
        self.h = np.random.random(self.num_dimensions)
        self.c = 0
        return self
    
    
    def initOptimizer(self,X,y):
        p = ng.p.Array(init = self.p,mutable_sigma=True)
        h = ng.p.Array(init = self.h,mutable_sigma=True)
        c = ng.p.Scalar(init = self.c,mutable_sigma=True)
        self.instrumentation = ng.p.Instrumentation(X,y,p,h,c)
        self.optimizer = ng.optimizers.NGOpt(
            parametrization=self.instrumentation,
            budget=self.buget,
            num_workers=self.n_workers
            )
        return self


    def loss(self,y_true,y_pred):
        return log_loss(y_true,y_pred)


    def fitness(self,X,y,p=None,h=None,c=None):
        if(type(p)==type(None)):
            p = self.p
        if(type(h)==type(None)):
            h = self.h
        if(type(c)==type(None)):
            c = self.c
        coeff = np.matmul(p,self.w)
        polys = []
        for i in range(self.num_dimensions):
            polynom = np.polynomial.Polynomial(coeff[i][::-1])
            polys.append(polynom)
        t_est = np.zeros((self.num_instances,self.num_dimensions))
        for i in range(self.num_instances):
            for j in range(self.num_dimensions):
                r = (polys[j] - X[i,j]).roots()
                r = r[r.imag==0]
                if(len(r)==0):
                    return np.inf
                else:
                    t_est[i,j] = r[0].real
        y_pred = np.matmul(t_est,h) + c
        y_pred = 1/(1+np.exp(-1*y_pred))
        return self.loss(y,y_pred)
    
    def fit(self,X,y):
        if(len(X.shape)==1):
            if(len(y)!=1):
                X = X.reshape(-1,1)
            else:
                X = X.reshape(1,-1)
        self.num_instances,self.num_dimensions = X.shape
        self.initWeights()
        self.initOptimizer(X, y)
        optimal_values = self.optimizer.minimize(self.fitness,verbosity=1)
        self.p = optimal_values.value[0][2]
        self.h = optimal_values.value[0][3]
        self.c = optimal_values.value[0][4]    
        print("Completed Model Training")
        return self

    def predict(self,X):
        if(len(X.shape)==1):
            if(len(y)!=1):
                X = X.reshape(-1,1)
            else:
                X = X.reshape(1,-1)
        coeff = np.matmul(self.p,self.w)
        polys = []
        for i in range(self.num_dimensions):
            polynom = np.polynomial.Polynomial(coeff[i][::-1])
            polys.append(polynom)
        t_est = np.zeros((self.num_instances,self.num_dimensions))
        for i in range(self.num_instances):
            for j in range(self.num_dimensions):
                r = (polys[j] - X[i,j]).roots()
                r_new = r[r.imag==0]
                if(len(r_new)==0):
                    t_est[i,j] = r[0].real
                else:
                    t_est[i,j] = r_new[0].real
        y_pred = np.matmul(t_est,self.h) + self.c
        y_pred = 1/(1+np.exp(-1*y_pred))
        return y_pred
    
    def evaluate(self,X,y_true):
        y_pred = self.predict(X)
        return self.loss(y_true,y_pred)


if __name__ == "__main__":
    model = BezierClassifier()
    X = np.ones((20,2))
    X[:,0] = np.arange(20)
    # X[:,1] = -1 * np.arange(20)
    y = np.ones(20)
    y[:5] = 0
    y[15:] = 0
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"Model Evaluation : {model.loss(y,y_pred)}")