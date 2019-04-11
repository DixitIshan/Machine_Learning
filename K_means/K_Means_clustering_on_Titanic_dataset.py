# IMPORTING ALL THE NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.cluster import KMeans
from sklearn import preprocessing

style.use('ggplot')

# READING IN THE DATASET
df = pd.read_excel('')

# EXPLORING THE DATASET
print(df.head())

# DROPPING UNNECESSARY COLUMNS
df.drop(['body', 'name', 'sex', 'boat' ], 1, inplace=True)

# REPLACING NULL VAULES WITH A 0
df.fillna(0, inplace=True)

# CREATING A FUNCTION TO HANDLE ALL THE NON NUMERIC DATASET
def handle_non_numeric(df):
	# EXTRACTING ALL THE COLUMN VALUES
	columns = df.columns.values
	# ITERATING THROUTH THOSE EXTRACTED VALUES
	for column in columns:
		digit_value = {}
		# SETTING UP THE THE PASSED PARAMETER AS KEY VALUE OF THE DICTIONARY
		def convert_to_int(val):
			return digit_value[val]

		# CHECKING IF ALL THE COLUMN VALUES ARE OF DATATYPE FLOAT OR INTEGER
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			# CONVERTING THE COLUMN TO A LIST OF ITS VALUES
			column_elements = df[column].values.tolist()
			# SETTING THE COLUMN TO FIND THE UNIQUE VALUES FROM IT
			unique_elements = set(column_elements)
			x = 0

			# SETTING ALL THE UNIQUE VALUES IN THE 'digit_value' AND MAPPING IT TO THE COLUMN VALUES
			for unique in unique_elements:
				if unique not in digit_value:
					digit_value[unique] = x
					x += 1
			
			df[column] = list(map(convert_to_int, df[column]))
	
	return df

df = handle_non_numeric(df)
print(df)


# CREATING A FEATURE ARRAY
X = np.array(df.drop(['survived'], 1).astype(float))

# SCALING THE DATASET
X = preprocessing.scale(X)

# CREATING A LABELS ARRAY
y = np.array(df['survived'])

# INVOKING THE KMEANS CLUSTERING ALGORITHM
clf = KMeans(n_clusters=2)
clf.fit(X)

# FOR EVERY CORRECT PREDICTION WE INCREASE THE VALUE OF THE COUNTER BY ONE
correct = 0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float64))
	predict_me = predict_me.reshape(-1, len(predict_me))
	prediction = clf.predict(predict_me)
	if prediction[0] == y[i]:
		correct += 1

print(correct/len(X))