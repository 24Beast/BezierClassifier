import numpy as np
import nevergrad as ng
from sklearn.metrics import log_loss
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

class BezierClassifier:

    def __init__(self,
                 max_points=8,
                 num_epochs=1000,
                 optimizer="ngopt",
                 num_workers=1,
                 verbosity=1):
        '''
        Parameters
        ----------
        max_points : Integer, optional
            Number of points being used to express the Bezier Curve. The default is 8.
        num_epochs : Integer, optional
            Number of Iterations for model fitting. The default is 1000.
        optimizer : string, optional
            Pass name of optimizer to be used. The default is "ngopt".
        num_workers : Integer, optional
            For parallel processing. The default is 1.
        verbosity : Integer, optional
            0 for silent, 1 for loss updates and 2 for loss and value updates. 
            The default is 1.

        Returns
        -------
        None.

        '''
        self.max_points = max_points
        self.epsilon = (0.1)**10
        self.xmin = None
        self.xmax = None
        self.num_instances = None
        self.num_dimensions = None
        self.buget = num_epochs
        self.show = verbosity
        self.n_workers = num_workers
        self.instrumentation = None
        self.opt_choice = optimizer
        self.optimizer = None
    
    def preprocess(self,X):
        '''
        Parameters
        ----------
        X : np.ndarray
            Preprocessing for training information. Involves shape correction
            and minmax scaling.

        Returns
        -------
        np.ndarray
            Normalized array of shape (num_items,num_features).

        '''
        if(len(X.shape)==1):
            if(len(y)!=1):
                X = X.reshape(-1,1)
            else:
                X = X.reshape(1,-1)
        if(type(self.xmin)==type(None)):
            min_x = np.min(X,axis=0)
            self.xmin = min_x
        else:
            min_x = self.xmin
        if(type(self.xmax)==type(None)):
            max_x = np.max(X,axis=0)
            self.xmax = max_x
        else:
            max_x = self.xmax
        max_x[min_x==max_x] += self.epsilon 
        return (X-min_x)/(max_x-min_x)
    
    
    def combin(self,n,i):
        '''
        Parameters
        ----------
        n : Integer
            N.
        i : Integer
            I.

        Returns
        -------
        Integer
            Combinatorial : nCi.

        '''
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
        if(self.opt_choice=="ngopt"):
            self.optimizer = ng.optimizers.NGOpt(
                parametrization=self.instrumentation,
                budget=self.buget,
                num_workers=self.n_workers
                )
        elif(self.opt_choice=="cma"):
            self.optimizer = ng.optimization.optimizerlib.CMA(
                parametrization=self.instrumentation,
                budget=self.buget,
                num_workers=self.n_workers
                )
        else:
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
                t_est[i,j] = polys[j](X[i,j])
        y_pred = np.matmul(t_est,h) + c
        y_pred = 1/(1+np.exp(-1*y_pred))
        return self.loss(y,y_pred)
    
    def fit(self,X,y):
        X = self.preprocess(X)
        self.num_instances,self.num_dimensions = X.shape
        self.initWeights()
        self.initOptimizer(X, y)
        optimal_values = self.optimizer.minimize(self.fitness,verbosity=self.show)
        self.p = optimal_values.value[0][2]
        self.h = optimal_values.value[0][3]
        self.c = optimal_values.value[0][4]    
        print("Completed Model Training")
        return self

    def predict(self,X):
        X = self.preprocess(X)
        coeff = np.matmul(self.p,self.w)
        polys = []
        for i in range(self.num_dimensions):
            polynom = np.polynomial.Polynomial(coeff[i][::-1])
            polys.append(polynom)
        t_est = np.zeros((X.shape[0],self.num_dimensions))
        for i in range(X.shape[0]):
            for j in range(self.num_dimensions):
                t_est[i,j] = polys[j](X[i,j])
        y_pred = np.matmul(t_est,self.h) + self.c
        y_pred = 1/(1+np.exp(-1*y_pred))
        return y_pred
    
    def evaluate(self,X,y_true):
        y_pred = self.predict(X)
        loss_val = self.loss(y_true,y_pred)
        print(f"Model Evaluation : {loss_val}")
        return loss_val
    


if __name__ == "__main__":
    from sklearn.metrics import classification_report as c_report
    model = BezierClassifier(max_points=8,
                             num_epochs=3000,
                             optimizer ="cma",
                             verbosity=0)
    X = np.ones((200,2))
    X[:,0] = np.arange(200)
    X[:,1] = -1 * np.arange(200)
    y = np.ones(200)
    y[:25] = 0
    y[75:125] = 0
    y[175:] = 0
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"Model Evaluation : {model.loss(y,y_pred)}")
    print(c_report(y, (y_pred>0.5).astype(np.uint8)))