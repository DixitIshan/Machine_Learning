# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import math
from collections import Counter

style.use('fivethirtyeight')


def KNN(data, predict, k = 4):
	if len(data) >= k:
		#WARNING THE USER FOR THE WRONG VALUE OF K
		warnings.warn("WARNINGS USER !!!")
	 # CREATING AN EMPTY LIST
	distances = []
	for group in data:
		for features in data[group]:
			# CALCULATING THE EUCLIDEAN DISTANCE BETWEEN ALL THE POINTS AND THE FEATURE POINT
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			# APPENDING THE RESULTS TO THE CREATED LIST
			distances.append([euclidean_distance, group])
			
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

# FALSIFIED TESTING DATASET
dataset = {'b':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}

# FEATURE TO PREDICT
new_features = [5,8]

# SCATTERING ALL THE POINTS FROM THE DATASET ON TO A MATPLOTLIB GRAPH
for i in dataset:
	for item in dataset[i]:
		plt.scatter(item[0], item[1],s = 50, color = i)

# INVOKING THE KNN FUNCTION AND STORING THE OUTPUT OF THE FUNCTION IN THE RESULT VARIABLES
result = KNN(dataset, new_features)

# PRINTTING THE RESULTS
print(result)

# PLOTTING THE SCATTERPLOT OF THE FEATURE
plt.scatter(new_features[0], new_features[1], s=100, color = result)
plt.show()