# IMPORT ALL THE NECESSARY LIBRARIES
import quandl
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime

#STYLE IS USED TO MAKE THE GRAPH APPEAR DECENT
style.use('ggplot')

#CREATING A DATAFRAME WITH GOOGLE STOCK DATA FROM QUANDL
df = quandl.get('WIKI/AAPl')

#DISCARDING UNNECESSARY COLUMS AND KEEPING ONLY THE RELEVANT COLUMN
df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low', 'Adj. Volume']]

#CALCULATING THE PERCENTAGE DIFFERENCE BETWEEN THAT DAY'S HIGH AND LOW
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0

#CALCULATING THE PERCENTAGE DIFFERENCE BETWEEN THAT DAY'S OPEN AND CLOSE
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

#RECREATING THE DATAFRAME
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#DEFINING A NEW VARIABLE WITH THE LABEL COLUMN
forecast_col = 'Adj. Close'

#FILLING ALL OF THE NAN VALUES WITH GARBAGE. THIS IS DONE IN ORDER TO RETAIN THE DATA AND NOT LOSE VALUABLE INFORMATION
df.fillna(value = -99999, inplace = True)

#NUMBER OF DAYS FOR PREDICTION(SHOULD BE IN INTEGER FORM)
forecast_out = int(math.ceil(len(df) * 0.01))

#IN THE LABEL COLUMN, SHIFTING THE OUTPUT BY 'FORECAST_OUT' NUMBERS INTO FUTURE
df['label'] = df[forecast_col].shift(-forecast_out)

'''
convectionally, for Machine Learning, uppercase x is used to represent features and lowercase y to represent labels
'''

#CONVERTING THE DATAFRAME OF FEATURES TO A NUMPY ARRAY AND DROPPING THE LABEL COLUMN
X = np.array(df.drop(['label'], 1))

#PREPROCESSING THE DATA BEFORE TRAINING
X = preprocessing.scale(X)

#X_Lately WILL BE ALL THE VALUES THAT ARE FROM THE NEGATIVE forecast_out
X_Lately = X[-forecast_out:]

#X WILL BE ALL THE VALUES UPTO THE NEGATIVE forecast_out
X = X[:-forecast_out]

#DROPPING ALL THEREMAINING  NAN VALUES AFTER SCALING
df.dropna(inplace = True)

#CONVERTING THE DATAFRAME OF LABELS TO A NUMPY ARRAY
y = np.array(df['label'])

# SPLITTING THE DATA INTO 80% - 20% RATIO. 80% FOR TRAINING X & Y AND 20% FOR TESTING X & Y
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

#INVOKING THE LINEAR REGRESSION METHOD
clf = LinearRegression()

#FITTING THE BEST FIT LINE INTO THE TRAINING DATASET(here 'fit' is synonyms with train)
clf.fit(X_train, y_train)

#CALCULATING THE CONFIDENCE SCORE AGAINST THE TESTING DATASET(here 'score' is synonyms with test)
confidence = clf.score(X_test, y_test)

#PREDICTING THE STOCK PRICES
forecast_set = clf.predict(X_Lately)

#POPULATING THE 'Forecast' COLUMN WITH NAN'S
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

#PLOTTING THE OUTPUT GRAPH
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()