#IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection, neighbors
from sklearn import metrics

style.use('fivethiryeight')

#READING IN THE DATASET
df = pd.read_csv('')

#PRINTING THE TOP 5 ROWS OF THE DATASET
df.head()

'''
STANDARDISE THE VARIABLES
'''

#SCALING IN KNN IS NECESSARY AS THE DISTANCE IS CALCULATED BETWEEN POINTS. THUS LARGE SCALE WILL HAVE A LARGE EFFECT ON THE DISTANCE BETWEEN THE OBSERVATIONS
scaler = preprocessing.StandardScaler()

#SCALING THE ENTIRE DATASET EXCEPT THE "TARGET CLASS" COLUMN
scaler.fit(df.drop(['TARGET CLASS'], 1))

#TRANSFORMING THE DATAFRAME INTO THE SCALED FEATURES OF ALL THE INDIVIDUAL VALUES
scaled_features = scaler.transform(df.drop(['TARGET CLASS'], 1))

#PRINTING THE SCALED_FEATURE
print(scaled_features)

#CREATING A NEW DATAFRAME OF ALL THE SCALED VALUED COLUMNS
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])

#PRINTING THE NEWELY FORMED DATAFRAME
print(df_feat)


'''
APPLYING KNN ALGORITHM
'''

#CREATING A 'FEATURE' DATAFRAME
X = df_feat

#CREATING A 'LABEL' DATAFRAME
y = df['TARGET CLASS']

# SPLITTING THE DATA INTO 80% - 20% RATIO. 80% FOR TRAINING X & Y AND 20% FOR TESTING X & Y
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

#INVOKING THE KNN CLASSIFIER
clf = neighbors.KNeighborsClassifier(n_neighbors = 1)

#CREATING A BEST FIT LINE ON THE TRAINING DATA SET
clf.fit(X_train, y_train)

#PREDICTING THE OUPUT VALUE ON THE TESTING DATASET
prediction = clf.predict(X_test)

#PRINTING THE PREDICTION
print(prediction)

#PRINTING THE CONFUSION MATRIX AND CLASSIFICATION REPORT OF THE TESTING DATASET
print(metrics.confusion_matrix(y_test, prediction))
print(metrics.classification_report(y_test, prediction))

'''
Choosing a K Value – Elbow method
'''

# CHECK ALL POSSIBLE ERROR VALUES FROM K=1 TO K=50 AND PLUG THE OUTPUT INTO THE EMPTY ARRAY ERROR_RATE

error_rate = []

#TRAINING THE CLASSIFIER AND CREATING A BESTFIT LINE FOR ALL THE VALUES, FROM 1 TO 50 OF, K
for item in range(1,50):
	knn = KNeighborsClassifier(n_neighbors = i)
	knn.fit(X_train, y_train)
	pred = knn.predict(X_test)
	error_rate.append(np.mean(pred != y_test))

#PLOTTING THE ERROR RATE GRAPH
plt.figure(flagsize = 10, 6)
plt.plot(range(1, 50), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')