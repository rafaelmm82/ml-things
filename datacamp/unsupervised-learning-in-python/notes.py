
# import packages
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

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

# analysing the inertia of models with number of cluster varing from 1 to 10
ks = range(1, 10)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(X)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# a dataframe named df with labels varing from 1 to 9, and companies names
# make a plot for each company name in a scatter plot considering the label values
import matplotlib.pyplot as plt

# Loop through each company
for company in df['companies'].unique():
    # Filter the data for the current company
    company_data = df[df['companies'] == company]
    
    # Scatter plot of label values for the current company, with company name mas text marker
    # plt.scatter(company_data['labels'], company_data['labels'], label=company, marker=company)
    plt.scatter(company_data['labels'], company_data['labels'], label=company, marker=company)
    # plt.scatter(company_data['labels'], company_data['labels'], label=company)


# Add title, legend, and labels
plt.title('Scatter Plot of Label Values for Each Company')
plt.legend()
plt.xlabel('Label')
plt.ylabel('Label')

# Show the plot
plt.show()

import matplotlib.pyplot as plt

# Loop through each company
for company in df['companies'].unique():
    # Filter the data for the current company
    company_data = df[df['companies'] == company]
    
    # Scatter plot of label values for the current company
    plt.scatter(company_data['labels'], company_data['labels'], label=company)
    
    # Add company name as text at each marker
    for label, x, y in zip(company_data['labels'], company_data['labels'], company_data['labels']):
        plt.text(x, y, company, ha='center', va='center')

# Add title, legend, and labels
plt.title('Scatter Plot of Label Values for Each Company')
plt.legend()
plt.xlabel('Label')
plt.ylabel('Label')

# Show the plot
plt.show()

