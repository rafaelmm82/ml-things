
# import packages
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
print(iris.feature_names)

# create and fit a kmeans model
model = KMeans(n_clusters=3)
model.fit(X)

print(model.labels_)
labels = model.predict(X)


# plot a 2d scatter plot for labels with cluster centroids
plt.scatter(X[:,0], X[:,1], c=labels, alpha=0.5)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='D', s=100)
plt.show()

plt.scatter(X[:,1], X[:,2], c=labels)
plt.scatter(model.cluster_centers_[:,1], model.cluster_centers_[:,2], marker='D', s=100)
plt.show()

plt.scatter(X[:,2], X[:,3], c=labels)
plt.scatter(model.cluster_centers_[:,2], model.cluster_centers_[:,3], marker='D', s=100)
plt.show()

plt.scatter(X[:,0], X[:,2], c=labels)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,2], marker='D', s=100)
plt.show()



