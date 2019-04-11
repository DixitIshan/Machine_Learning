# IMPORTING ALL THE NECESSARY LIBRARIES AND MODULES
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

style.use('ggplot')

# CREATING A DATSET
data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8)

# EXPLORING THR DATSET
print(data)

# PLOTTING A SCATTERPLOT GRAPH OF THE DATASET
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

# INVOKING THE K_MEANS CLUSTERING ALGORITHM FROM SKLEARN
KMeans = KMeans(n_clusters = 4)

# CREATING A BEST FIT ON THE DATASET
KMeans.fit(data[0])

# PRINTING THE CLUSTER CENTERS AND LABELS OF THE DATASET
print(KMeans.cluster_centers_)
print(KMeans.labels_)

# PLOTTING THE SCATTERPLOT GRAPH OF THE ORIGINAL AND THE CLUSTERED DATA ON THE SAME AXIS
fig, (ax1, ax2) = plt.subplot(1, 2, sharey = True, figsize = (10, 6))

ax1.set_title('KMeans')
ax1.scatter(data[0][:,0], data[0][:,1], c = kmeans.labels_, cmap = 'rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow')