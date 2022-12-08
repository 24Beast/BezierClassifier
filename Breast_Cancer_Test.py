# Importing Libraries
import numpy as np
from Bezier import BezierClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Loading Data
data = load_breast_cancer()
X,y = data.data,data.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

# Training Model
model = BezierClassifier(max_points=8,
                         num_epochs=1500,
                         optimizer ="cma",
                         verbosity=0)
model.fit(X_train,y_train)
model.evaluate(X_train,y_train)

# Testing Model
model.evaluate(X_test,y_test)
y_pred = model.predict(X_test)
y_pred = (y_pred>0.5).astype(np.uint8)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))