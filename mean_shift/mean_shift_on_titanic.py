# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation
import pandas as pd
import matplotlib.pyplot as plt

# READING IN THE DATASET
df = pd.read_excel('titanic.xls')

original_df = pd.DataFrame.copy(df)

# DROPPING IN THE UNNECESSARY COLUMNS
df.drop(['body', 'name', 'ticket', 'home.dest'], 1, inplace=True)
# DEALING WITH NAN's
df.fillna(0,inplace=True)

# DEALING WITH NON NUMERICAL VALUES
df = pd.get_dummies(df, columns = '')

#DIVIDING THE DATA IN FEATURES AND LABEL.
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

# INVOKING THE CLUSTER
clf = MeanShift()
# FITTING THE DATA INTO THE CLUSTERING ALGORITHM
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# ADDING A NEW COLUMN TO OUR ORIGINAL DATAFRAME
original_df['cluster_group']=np.nan

# POPULATING THE NEWELY ADDED COLUMNS WITH THE LABELS
for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]


#
n_clusters_ = len(np.unique(labels))
survival_rate = {}

for i in range(n_clusters_):
	# GETTING ONLY THE CLUSTER GROUP WHICH HAS THE OF N_CLUSTERS_
	temporary_df = original_df[(original_df['cluster_group'] == float(i))]
	print(temporary_df)

	# GETTING THE CLUSTER WITH ONLY THE SURVIVORS
	survival_cluster = temporary_df[(temporary_df['survived'] == 1)]

	survival_rate = len(survival_cluster) / len(temporary_df)
	survival_rate[i] = survival_rate

print(survival_rate)
