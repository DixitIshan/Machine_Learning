#IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#STYLING TO MAKE THE GRAPH APPEAR PRESENTABLE
style.use('ggplot')

#READING IN THE CSV THAT CONTAINS ALL THE DATA
df = pd.read_csv('')

#DESCRIBING AND GAINING INSIGHTS ON THE DATASET
df.head()
df.info()
df.describe()

#PLOTTING THE SEABORN GRAPHS OF DATASET, PRICE AND CORRELATION
sns.pairplot(df)
sns.distplot(df['price'])
sns.heatplot(df.corr(), annot = True)

#PRINTING THE COLUMNS
print(df.columns)

#DIVIDING THE DATA IN FEATURES AND LABEL.
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = df[['Price']]

#SPLITTING THE DATA INTO TRAINING AND TESTING DATA BATCHES(60-40 RATIO)
X_train, X_test, y_train,y_test = model_selection.train_test_split(X, y, test_size = 0.4, random_state = 101)

#INVOKING THE LINEAR REGRESSION MODEL
clf = LinearRegression()

#FITTING THE LINE FOR THE TEAINGING DATASET
clf.fit(X_train, y_train)

'''
here
if y(w, x) = w0 + w1x1 + w2x2 + ... + wpxp
then
w0 is the intercept_ and (w1, w2, w3, ... wp) are the coef_
'''
#PRINTING THE INTERCEPT AND THE COEFFICIENT
print(clf.intercept_)
print(clf.coef_)

# CREATING A NEW DATAFRAME TO VISUALISE ALL THE CHANGES IN ANY SPECIFIC FEATURE WHEN ALL OTHER FEATURES ARE STAGNANT
cdf = pd.DataFrame(data = clf.coef_, index = X.columns, columns = ['Coeff'])

#PREDICTING FOR THE TESTING DATA SET
predictions = clf.predict(X_test)

#PLOTTING A SCATTERPLOT FOR PREDICTIONS
plt.scatter(y_test, predictions)

PLOTTING A DISTPLOT OF 
sns.distplot((y_test - predictions))

'''
Three common evaluation metrics for regression problems (errors = actual y – predicted y):

Mean Absolute Error (MAE) is the mean of the absolute value of the errors
Mean Squared Error (MSE) is the mean of the squared errors
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors

Comparing these metrics:

MAE is the easiest to understand, because it’s the average error.
MSE is more popular than MAE, because MSE “punishes” larger errors, which tends to be useful in the real world.
RMSE is even more popular than MSE, because RMSE is interpretable in the “y” units.
All of these are loss functions, because we want to minimize them!
'''
#PRINTING OF ALL THE ERRORS IN OUR PREDICTIONS
print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))