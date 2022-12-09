# Importing Libraries
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from Bezier import BezierClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


thismodule = sys.modules[__name__]


# Data Functions
def toyData1(X): # 6, 1500
    y = np.zeros(len(X))
    f1_mid,f2_mid = ((X.max(axis=0) - X.min(axis=0))/2) + X.min(axis=0)
    y[(X[:,0]>f1_mid) * (X[:,1]>f2_mid)] = 1
    y[(X[:,0]<f1_mid) * (X[:,1]<f2_mid)] = 1
    return y

def toyData2(X):# 4, 1500
    y = np.zeros(len(X))
    f1_mid,f2_mid = ((X.max(axis=0) - X.min(axis=0))/2) + X.min(axis=0)
    y[np.sum((X-[f1_mid,f2_mid])**2,axis=1)<1000] = 1
    return y

def toyData3(X): # 8, 4000
    y = np.zeros(len(X))
    y[(X[:,0]<30) * (X[:,0]>20)] = 1
    y[(X[:,0]<80) * (X[:,0]>60)] = 1
    return y

def toyData4(X): # 4, 2000
    y = np.zeros(len(X))
    y[(X[:,0]<70) * (X[:,0]>30) * (X[:,1]<60) * (X[:,1]>30)] = 1
    return y



# Display Function
def showData(X,y,title):
    plt.scatter(X[:,0],X[:,1],c=y,cmap="jet")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()


degrees = [8,4,8,4]
n_epochs = [2000,2000,4000,2000]

# Data Generation
X = np.array([[i,j] for i in range(100) for j in range(100)]).astype(np.float32)

for i in range(1,5):
    datafunc = getattr(thismodule, f"toyData{i}")
    y = datafunc(X)
    showData(X,y,f"Experimental Data {i}")

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0)
    
    # Training Bezier Model
    model = BezierClassifier(max_points=degrees[i-1],
                             num_epochs=n_epochs[i-1],
                             optimizer ="ngopt",
                             verbosity=0)
    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print(f"Bezier Time : {end-start}")
    
    # Testing Bezier Model
    y_pred = model.predict(X_test)
    y_pred = (y_pred>0.5).astype(np.uint8)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    showData(X,model.predict(X),"Bezier")
    
    
    # Training Logistic Model
    model = LogisticRegression()
    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print(f"Logistic Time : {end-start}")
    
    
    # Testing Logistic Model
    y_pred = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred>0.5).astype(np.uint8)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    showData(X,model.predict_proba(X)[:,1],"Logistic Regression")
    
    
    # Training Linear SVM Model
    model = SVC(kernel="linear",probability=True)
    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print(f"Linear SVM Time : {end-start}")
    
    
    # Testing Linear SVM Model
    y_pred = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred>0.5).astype(np.uint8)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    showData(X,model.predict_proba(X)[:,1],"Linear SVM")
    
    
    # Training RBF SVM Model
    model = SVC(kernel="rbf",probability=True)
    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print(f"RBF SVM Time : {end-start}")
    
    
    # Testing RBF SVM Model
    y_pred = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred>0.5).astype(np.uint8)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    showData(X,model.predict_proba(X)[:,1],"RBF Kernel SVM")
    
    '''
    # Training Poly SVM Model
    model = SVC(kernel="poly",probability=True,degree = min(4,degrees[i-1]))
    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print(f"Poly SVM Time : {end-start}")
    
    
    # Testing Poly SVM Model
    y_pred = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred>0.5).astype(np.uint8)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    showData(X,model.predict_proba(X)[:,1],"Poly Kernel SVM")
    '''
    
    # Training Decision Tree Model
    model = DecisionTreeClassifier()
    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print(f"Decision Tree Time : {end-start}")
    
    
    # Testing Decision Tree Model
    y_pred = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred>0.5).astype(np.uint8)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    showData(X,model.predict_proba(X)[:,1],"Decision Tree")
