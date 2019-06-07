import math
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

class mykmeans(object):

    def __init__(self, k=3, max_iter=200, eps=0.01):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
      
    def get_k(self):
        """
        Return number of clusters
        """
        return self.k
    
    def get_max_iter(self):
        """
        Return maximum number of iterations
        """
        return self.max_iter
    
    def get_eps(self):
        """
        Return tolerance value
        """
        return self.eps

    def euclidean_dist(self, X, Y):
        """
        Compute Euclidean distance between arrays
        Input:
            - X: (1,d) array or (n,d) array
            - Y: (1,d) array or (n,d) array
        """
        dist = np.linalg.norm(X - Y)
        return dist
    
    def fit(self, X):
        """
        K-Means clustering algorithm
        Input:
            - X: (n,d) array
        """

        k = self.get_k()
        max_iter = self.get_max_iter()
        eps = self.get_eps()
        compdist = self.euclidean_dist
       
        n, d = X.shape
        
        # Initialize k centroids randomly generated within the data
        np.random.seed(0)
        idx = np.random.randint(0, n, k)
        self.centers = deepcopy(X[idx])
        
        # Initialize labels
        self.labels = np.zeros(n)
        
        # Clustering
        t = 0
        while t < max_iter:
            # Initiate clusters
            self.clusters = {}
            for i in range(k):
                self.clusters[i] = []
        
            # Associate each data point with the nearest centroid
            for i in range(len(X)):
                point = X[i]
                dists = []
                
                # Compute distance between each point and centroids
                for center in self.centers:
                    dist = compdist(point, center)
                    dists.append(dist)

                # Assign the nearest centroid
                label = np.argmin(dists)
                self.clusters[label].append(point)
                
                # Assign/Update data labels
                self.labels[i] = label
            
            # Save previous centroids
            centers_prev = deepcopy(self.centers)
            
            # Update centroid 
            for label in self.clusters:
                if len(self.clusters[label]) != 0:
                    self.centers[label] = np.mean(self.clusters[label], axis=0)
                
            # Stop iterations once converges
            if compdist(self.centers, centers_prev) <= eps:
                break
            
            t += 1
        self.num_iter = t
