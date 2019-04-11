# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.cluster import KMeans
style.use('ggplot')

# FALSIFIED DATASET
X = np.array([[1, 2],[1.5, 1.8],[5, 8],[8, 8],[1, 0.6],[9, 11]])

# INVOKING THE KMEANS CLUSTERING ALGORITHM
Kmeans = KMeans(n_clusters = 2)

# CREATING A BEST FIT ON THE DATASET
Kmeans.fit(X)

# THESE ARE THE CENTROIDS AND THE LABELS
centroids = Kmeans.cluster_centers_
print(centroids)
labels = Kmeans.labels_
print(labels)

# ITERATING THROUGH THE DATASET AND PLOTTING A SCATTERPLOT GRAPH OF DATASET AND CENTROIDS
colors = ["g.","r.","c.","y."]
for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()