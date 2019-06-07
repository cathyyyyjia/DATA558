import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from mykmeans import mykmeans



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
print('\n======================== Simulated Dataset Demo ========================')
print('\nSimulated dataset')
print('Number of observations: 300')
print('Number of features: 20')
print('Labels:', labels_name)



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
                      marker='o', color=colors[label], label=label)
for center in mycluster.centers:
    txt = plt.scatter(center[0], center[1], marker='x', color='tab:red', s=100)
txt = plt.scatter([], [], marker='x', color='tab:red', s=100, label='centroids')
txt = plt.title('K-Means Clustering Result', fontsize=15)
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

print('\nLabels clustered by my k-means clustering\n', mylabels)
print('\nDataset labels\n', labels)
match_labels = np.zeros(mylabels.shape)
match_labels[mylabels==0] = 2
match_labels[mylabels==1] = 0
match_labels[mylabels==2] = 1
print('\nMatching labels with dataset labels...')
print('\nMatched labels clustered by my k-means clustering\n', match_labels.astype(int))

# Compute accuracy

score = np.mean(match_labels==labels) * 100
print("\nAccuracy of my k-means clustering = %.2f%%\n" % score)
