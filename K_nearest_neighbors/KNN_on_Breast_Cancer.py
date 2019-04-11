#IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

#READING IN THE DATASET CSV
df = pd.read_csv('')

#REPLACING ALL THE NULL/FALSE VALUES BY -99999.
df.replace('?', -99999, inplace= True)

#DROPPING THE ID COLUMN AS IT CAN BE HARMFUL FOR THE ALGORITHM
df.drop(['id'], 1, inplace = True)

'''
convectionally, for Machine Learning, uppercase x is used to represent features and lowercase y to represent labels
'''
#CONVERTING THE DATAFRAME OF FEATURES TO A NUMPY ARRAY AND DROPPING THE CLASS COLUMN (FEATURE)
X = np.array(df.drop['class'], 1)

#CONVERTING THE DATAFRAME OF CLASS TO A NUMPY ARRAY(LABEL)
y = np.array(df['class'])

# SPLITTING THE DATA INTO 80% - 20% RATIO. 80% FOR TRAINING X & Y AND 20% FOR TESTING X & Y
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

#INVOKING THE KNN CLASSIFIER FROM SKLEARN
clf = neighbors.KNeighborsClassifier()

#CREATING A BESTFIT LINE
clf.fit(X_train, y_train)

#MEASURING THE ACCURACY OF THE TRAINED MODEL ON THE TESTING DATASET
accuracy = clf.score(X_test, y_test)

#PRINTING THE ACCURACY OF THE TRAINED MODEL 
print(accuracy)

#CREATING A FALSIFIED DATA TO CHECK OUR ALGORITHMS EFFICIENCY
example_measure = np.array([4,2,1,1,1,2,3,2,1])

#RESHAPING THE CREATED DATA INTO A PROPER FEED FORMAT
example_measure = example_measure.reshape(len(example_measure), -1)

#PREDICTING THE OUTPUT OF OUR OWN CREATED DATA
prediction = clf.predict(example_measure)

#PRINTING THE PREDICTION OUTPUT
print(prediction)