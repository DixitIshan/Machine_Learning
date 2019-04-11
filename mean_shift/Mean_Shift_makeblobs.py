# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')

# DEFINING THE CENTERS
centers = [[1,1,1],[5,5,5],[3,10,10]]

# CREATING THE DATASET OF 100 BLOB VALUES
X, _ = make_blobs(n_samples = 100, centers = centers, cluster_std = 1.5)

# INVOKING THE MEAN SHIFT CLUSTERING ALGORITHM FROM SKLEARN
clst = MeanShift()
# FITTING THE DATA
clst.fit(X)

# GETTING THE LABELS AND CLUSTER CENTERS
labels = clst.labels_
cluster_centers = clst.cluster_centers_

print(cluster_centers)

n_clusters = len(np.unique(labels))

print('Number of estimated clusters: ', n_clusters)

# PLOTTING THE 3D SCATTERPLOT GRAPH
colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

for i in range(len(X)):
	ax.scatter(X[i][0], X[i][1], c = colors[labels[i]], marker = 'o')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:,2], marker="x",color='k', s=150, linewidths = 5, zorder=10) 

plt.show()