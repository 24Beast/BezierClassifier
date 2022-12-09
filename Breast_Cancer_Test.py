# Importing Libraries
import time
import numpy as np
from sklearn.svm import SVC
from Bezier import BezierClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



# Loading Data
data = load_breast_cancer()
X,y = data.data,data.target

# Train Test Split
kf = KFold(n_splits=5)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Training Model
    model = BezierClassifier(max_points=8,
                             num_epochs=1500,
                             optimizer ="cma",
                             verbosity=0)
    start = time.time()
    model.fit(X_train,y_train)
    end = time.time()
    print(f"Bezier Time : {end-start}")
    model.evaluate(X_train,y_train)
    
    # Testing Model
    model.evaluate(X_test,y_test)
    y_pred = model.predict(X_test)
    y_pred = (y_pred>0.5).astype(np.uint8)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

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
        
    
    # Training Decision Model
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
    
