import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from mykmeans import mykmeans



print('\n======================== Experimental Comparison ========================')



print('\n--------------------------- Simulated Dataset ---------------------------')

### Simulate Dataset ###

# Simulate a dataset with size = 300, dimension = 20

np.random.seed(0)
features = np.random.normal(loc=0, scale=1, size=(300,20))
features[100:200,] = np.random.normal(loc=5, scale=1, size=(100,20))
features[200:300,] = np.random.normal(loc=10, scale=1, size=(100,20))
labels = np.zeros(300)
labels[100:200,] = 1
labels[200:300,] = 2
labels = labels.astype(int)
labels_name = [0, 1, 2]

### Clustering ###

# Apply K-means clustering using mykmeans

print('\nApplying K-means clustering using mykmeans...\nNumber of clusters = %d' % 3)
mycluster = mykmeans(k=3)
mycluster.fit(features)
mylabels = mycluster.labels.astype(int)

# Apply K-means clustering using sklearn kmeans

print('\nApplying K-means clustering using sklearn kmeans...\nNumber of clusters = %d' % 3)
skcluster = KMeans(n_clusters=3, random_state=0)
skcluster.fit(features)
sklabels = skcluster.labels_.astype(int)

### Visualization ###

# My k-means clustering

txt = plt.figure(figsize=(8, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for label in mycluster.clusters:
    txt = plt.scatter(features[mylabels==label][:,0], features[mylabels==label][:,1],
                      marker='o', color=colors[label], label='Cluster '+str(label))
for center in mycluster.centers:
    txt = plt.scatter(center[0], center[1], marker='x', color='tab:red', s=100)
txt = plt.scatter([], [], marker='x', color='tab:red', s=100, label='Centroids')
txt = plt.title('My K-Means Clustering', fontsize=15)
txt = plt.xlabel('Feature 1', fontsize=12)
txt = plt.ylabel('Feature 2', fontsize=12)
txt = plt.legend()
plt.show()

# sklean k-means clustering

txt = plt.figure(figsize=(8, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for label in np.unique(sklabels):
    txt = plt.scatter(features[sklabels==label][:,0], features[sklabels==label][:,1],
                      marker='o', color=colors[label], label='Cluster '+str(label))
for center in skcluster.cluster_centers_:
    txt = plt.scatter(center[0], center[1], marker='x', color='tab:red', s=100)
txt = plt.scatter([], [], marker='x', color='tab:red', s=100, label='Centroids')
txt = plt.title('Sklearn K-Means Clustering', fontsize=15)
txt = plt.xlabel('Feature 1', fontsize=12)
txt = plt.ylabel('Feature 2', fontsize=12)
txt = plt.legend()
plt.show()

# Dataset labels

txt = plt.figure(figsize=(8, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(len(labels_name)):
    txt = plt.scatter(features[labels==i][:,0], features[labels==i][:,1],
                      marker='o', color=colors[i], label=i)
txt = plt.title('Dataset Labels', fontsize=15)
txt = plt.xlabel('Feature 1', fontsize=12)
txt = plt.ylabel('Feature 2', fontsize=12)
txt = plt.legend()
plt.show()

### Performance ###

# Match labels with dataset labels

match_mylabels = np.zeros(mylabels.shape)
match_mylabels[mylabels==0] = 2
match_mylabels[mylabels==1] = 0
match_mylabels[mylabels==2] = 1
print('\nMatching labels with dataset labels...')
print('\nMatched labels clustered by my k-means clustering\n', match_mylabels.astype(int))

# Compute accuracy
score = np.mean(match_mylabels==labels) * 100
print("\nAccuracy of my k-means clustering = %.2f%%" % score)

match_sklabels = np.zeros(sklabels.shape)
match_sklabels[sklabels==0] = 1
match_sklabels[sklabels==1] = 2
match_sklabels[sklabels==2] = 0
print('\nMatched labels clustered by sklearn k-means clustering\n', match_sklabels.astype(int))

# Compute accuracy
score = np.mean(match_sklabels==labels) * 100
print("\nAccuracy of my k-means clustering = %.2f%%" % score)



print('\n--------------------------- Real-World Dataset ---------------------------')

### Load Real-World Dataset ###

# Load iris dataset

iris = datasets.load_iris()
features = iris.data
labels = iris.target

### Apply My K-Means Clustering ###

# Apply K-means clustering using mykmeans

print('\nApplying K-means clustering using mykmeans...\nNumber of clusters = %d' % 3)
mycluster = mykmeans(k=3)
mycluster.fit(features)
mylabels = mycluster.labels.astype(int)

# Apply K-means clustering using sklearn kmeans

print('\nApplying K-means clustering using sklearn kmeans...\nNumber of clusters = %d' % 3)
skcluster = KMeans(n_clusters=3, random_state=0)
skcluster.fit(features)
sklabels = skcluster.labels_.astype(int)

### Visualization ###

# My k-means clustering

txt = plt.figure(figsize=(8, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for label in mycluster.clusters:
    txt = plt.scatter(features[mylabels==label][:,0], features[mylabels==label][:,1],
                      marker='o', color=colors[label], label='Cluster '+str(label))
for center in mycluster.centers:
    txt = plt.scatter(center[0], center[1], marker='x', color='tab:red', s=100)
txt = plt.scatter([], [], marker='x', color='tab:red', s=100, label='Centroids')
txt = plt.title('My K-Means Clustering', fontsize=15)
txt = plt.xlabel(iris.feature_names[0], fontsize=12)
txt = plt.ylabel(iris.feature_names[1], fontsize=12)
txt = plt.legend()
plt.show()

# sklean k-means clustering

txt = plt.figure(figsize=(8, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for label in np.unique(sklabels):
    txt = plt.scatter(features[sklabels==label][:,0], features[sklabels==label][:,1],
                      marker='o', color=colors[label], label='Cluster '+str(label))
for center in skcluster.cluster_centers_:
    txt = plt.scatter(center[0], center[1], marker='x', color='tab:red', s=100)
txt = plt.scatter([], [], marker='x', color='tab:red', s=100, label='Centroids')
txt = plt.title('Sklearn K-Means Clustering', fontsize=15)
txt = plt.xlabel(iris.feature_names[0], fontsize=12)
txt = plt.ylabel(iris.feature_names[1], fontsize=12)
txt = plt.legend()
plt.show()

# Dataset labels

txt = plt.figure(figsize=(8, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i in range(len(iris.target_names)):
    txt = plt.scatter(features[labels==i][:,0], features[labels==i][:,1],
                      marker='o', color=colors[i], label=iris.target_names[i])
txt = plt.title('Dataset Labels', fontsize=15)
txt = plt.xlabel(iris.feature_names[0], fontsize=12)
txt = plt.ylabel(iris.feature_names[1], fontsize=12)
txt = plt.legend()
plt.show()



### Performance ###

# Match labels with dataset labels

match_mylabels = np.zeros(mylabels.shape)
match_mylabels[mylabels==1] = 2
match_mylabels[mylabels==2] = 1
print('\nMatching labels with dataset labels...')
print('\nMatched labels clustered by my k-means clustering\n', match_mylabels.astype(int))

# Compute accuracy
score = np.mean(match_mylabels==labels) * 100
print("\nAccuracy of my k-means clustering = %.2f%%" % score)

match_sklabels = np.zeros(sklabels.shape)
match_sklabels[sklabels==0] = 1
match_sklabels[sklabels==1] = 0
match_sklabels[sklabels==2] = 2
print('\nMatched labels clustered by sklearn k-means clustering\n', match_sklabels.astype(int))

# Compute accuracy
score = np.mean(match_sklabels==labels) * 100
print("\nAccuracy of sklearn k-means clustering = %.2f%%\n" % score)
