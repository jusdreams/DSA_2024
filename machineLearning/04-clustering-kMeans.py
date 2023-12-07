'''
Let’s learn how to divide a set of customers into categories, based on characteristics
they share. One of the most adopted approaches that can
be used for customer segmentation is clustering. Clustering can group data only “unsupervised,”
based on the similarity of customers to each other.
It will partition your customers into mutually exclusive groups, for example, into 3 clusters.

This is a step further than classification, now we will take in multiple similarities and build groups of data based off their classification similarities.



K-Means Clustering plus Advantages & Disadvantages 
Hierarchical Clustering plus Advantages & Disadvantages 
Measuring the Distances Between Clusters - Single Linkage Clustering 
Measuring the Distances Between Clusters - Algorithms for Hierarchy Clustering 
Density-Based Clustering
'''

'''
k-Means on a randomly generated dataset
    Setting up K-Means
    Creating the Visual Plot

Customer Segmentation with K-Means
    Pre-processing
    Modeling
    Insights
'''
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random 
import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
#%matplotlib inline
matplotlib.use('inline')

#load data
np.random.seed(0) #creating our own dataset
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9) #make_blobs makes random cluster of points
plt.scatter(X[:,0], X[:,1], marker='.') #display scatter plot of random data

'''
The KMeans class has many parameters that can be used, but we will be using these three:

    init: Initialization method of the centroids.
        Value will be: "k-means++"
        k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
    n_clusters: The number of clusters to form as well as the number of centroids to generate.
        Value will be: 4 (since we have 4 centers)
    n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        Value will be: 12

Initialize KMeans with these parameters, where the output parameter is called k_means.
'''
#set up kmeans
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12) #intialize

k_means.fit(X) #fit

k_means_labels = k_means.labels_ #grab labels
k_means_labels

k_means_cluster_centers = k_means.cluster_centers_ #get coordinates of cluster center
k_means_cluster_centers

#visual plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

#add customer data
cust_df = pd.read_csv('/Users/jusdreams/Documents/GitHub/2024/DSA/data/Cust_Segmentation.csv')
df = cust_df.drop('Address', axis=1)
df.head()

# normalize dataset
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

#Modeling
#apply k-means on our dataset, and take a look at cluster labels.
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

df["Clus_km"] = labels #assign labels to each row in dataframe
df.head(5)

#check the centroid values by averaging features in cluster
df.groupby('Clus_km').mean()

#look at distribution
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))