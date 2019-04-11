# IMPORTING ALL THE NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# IMPORTING THE DATASET
from sklearn import datasets, preprocessing, model_selection, decomposition, neighbors
style.use('ggplot')

# LOADING IN THE DATASET
iris = datasets.load_iris()

# CREATING A FEATURE DATAFRAME
X = pd.DataFrame(iris['data'], columns = iris['feature_names'])
# CREATING A LABEL DATAFRAME
y = iris['target']

# PRINTING OUT THE HEAD
X.head()

# CREATING A PREPROCESSING SCALER
scaler = preprocessing.MinMaxScaler()

# FITTING THE DATASET INTO THE SCALER
X = scaler.fit_transform(X)

# SPLITTING THE DATA INTO TRAINING AND TESTING DATASETS
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_State = 1)

# CREATING A PRINCIPAL COMPONENT ANALYSIS MODEL ON TRAINING AND TESTING DATASET
pca_model = decomposition.PCA(n_components = 2)

# FITTING THE TRAINING DATASET INTO THE PCA MODEL
pca_model.fit(X_train)

# TRANSFORMING THE TRAINING AND TESTING DATASET INTO PCA MODELS
X_train = pca_model.transform(X_train)
X_test = pca_model.transform(X_test)

# NUMBER OF RANNDON CENTROIDS
k = 5

# INVOKING THE KNN CLASSIFIER
clf = neighbors.KNeighborsClassifier(n_neighbors = k)

# CREATING A BEST FIT ON THE TRAINING DATASET
clf.fit(X_train, y_train)

# PREDICTING ON THE TESTING DATASET
pred = clf.predict(X_test)

# PRINTING THE OUTPUT
print(pred)