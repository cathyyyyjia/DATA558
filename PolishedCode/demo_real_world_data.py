import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from mykmeans import mykmeans



### Load Real-World Dataset ###

# Load iris dataset

iris = datasets.load_iris()
features = iris.data
labels = iris.target
print('\n======================== Real-World Dataset Demo ========================')
print('\nDataset: Iris')
print('Features:', iris.feature_names)
print('Labels:', iris.target_names)



### Apply My K-Means Clustering ###

# Apply K-means clustering using mykmeans

print('\nApplying K-means clustering using mykmeans...\nNumber of clusters = %d' % 3)
mycluster = mykmeans(k=3)
mycluster.fit(features)
mylabels = mycluster.labels.astype(int)



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

print('\nLabels clustered by my k-means clustering\n', mylabels)
print('\nDataset labels\n', labels)
match_labels = np.zeros(mylabels.shape)
match_labels[mylabels==1] = 2
match_labels[mylabels==2] = 1
print('\nMatching labels with dataset labels...')
print('\nMatched labels clustered by my k-means clustering\n', match_labels.astype(int))

# Compute accuracy

score = np.mean(match_labels==labels) * 100
print("\nAccuracy of my k-means clustering = %.2f%%\n" % score)
