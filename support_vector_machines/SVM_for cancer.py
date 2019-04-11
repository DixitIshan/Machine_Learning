#IMPORTING ALL THE BASIC NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
#IMPORTING THE DATASET FROM SCIKIT-LEARN
from sklearn.datasets import load_breast_cancer
from sklearn import model_selection
#IMPORTING THE SUPPORT VECTOR CLASSIFIER
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
#IMPORTING THE GRID SEARCH CROSS VALIDATOR TO FIND THE BEST FIT PARAMETERS FOR OPIMAL RESULTS ON TESTING DATASET
from sklearn.grid_search import GridSearchCV

style.use('fivethirtyeight')

#LOADING IN THE DATASET
cancer = load_breast_cancer()

#DESCRIBING THE DATASET
cancer.keys()

#CREATING A PANDAS DATAFRAME
df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

#EXPLORINT THE DATASET
df_feat.head()

#DIVIDING THE DATA IN FEATURES AND LABEL.
X = df_feat
y = cancer['target']

#SPLITTING THE DATA INTO TRAINING AND TESTING DATA BATCHES(60-40 RATIO)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

##INVOKING THE SUPPORT VECTOR CLASSIFIER MODEL
clf = SVC()

#FITTING THE LINE FOR THE TRAINING DATASET
clf.fit(X_train, y_train)

#MAKING THE PREDICTIONS ON THE TESTING DATASET
predictions = clf.predict(X_test)

#EXPLORING THE CONFUSION_MATRIX & CLASSIFICATION_REPORT
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

'''
Finding the right parameters (like what C or gamma values to use) is a tricky task! We can try a bunch of combinations and see what works best!
This idea of creating a ‘grid’ of parameters and just trying out all the possible combinations is called a Gridsearch.
Scikit-learn has this functionality built in with GridSearchCV! (CV = cross validation)
GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train.
The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested.

C controls the costs of misclassification on the training data – high C value gives you low bias and high variance as there is a high penalisation for misclassification
Large gamma will lead to high bias and low variance
'''
grid_parameters = {'C' : [0.1,1, 10, 100, 1000], 'gamma' : [1,0.1,0.01,0.001,0.0001], 'kernel' : [rbf]}
grid = GridSearchCV(SVC(), grid_parameters, verbose = 10)
grid.fit(X_train,y_train)

#TO GET THE BEST PARAMETERS AND ESTIMATORS OF OUR NEW MODEL
print(grid.best_params_)
print(grid.best_estimator_)

#PREDICTING BASED ON OUR NEW MODEL
prediction_with_grid = grid.predict(X_test)

#EXPLORING THE CONFUSION_MATRIX & CLASSIFICATION_REPORT OF OUR NEW MODEL ON THE TEST DATA
print(confusion_matrix(y_test, prediction_with_grid))
print(classification_report(y_test,prediction_with_grid))