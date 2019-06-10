from copy import deepcopy
import numpy as np
import scipy

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

class myspectral(object):

    def __init__(self, k, sigma=1):
        self.k = k
        self.sigma = sigma

    def get_k(self):
        """
        Return number of clusters
        """
        return self.k
    
    def get_sigma(self):
        """
        Return sigma for computing kernel
        """
        return self.sigma

    def rbf(self, X, Y, sigma):
        """
        Compute RBF kernel
        Input:
            - X: array
            - Y: array
            - sigma: Tuning parameter
        """
        if Y.shape[1] == 2:
            return np.exp(-np.linalg.norm(np.subtract(X[:, :, np.newaxis], Y[:, :, np.newaxis].T), axis=1) ** 2 /
                          (2*sigma**2))
        else:
            return np.exp(-np.linalg.norm(np.subtract(X, Y), axis=1) ** 2 / (2*sigma ** 2))
    
    def kernel(self, X, sigma):
        """
        Compute kernel
        Input:
            - X: array
            - sigma: Tuning parameter
        """
        return self.rbf(X, X, sigma)
        
    def laplacian(self, X, sigma=1):
        """
        L = G - W
        Input:
            - X: (n,n) array
            - sigma: Tuning parameter
        """
        # W is the adjacency matrix
        n = len(X)
        W = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    W[i,j] = 0
                else:
                    W[i,j] = np.exp((-np.linalg.norm(X[i]-X[j])**2)/(2*sigma**2))
        # G is the diagonal matrix with node degrees
        D = np.sum(W, axis=0)
        G = np.diag(D)
        return G - W
    
    def fit(self, X):
        """
        Spectral clustering algorithm
        Input:
            - X: (n,d) array
        """
        k = self.get_k()
        sigma = self.get_sigma()
        
        # Compute RBF kernel
        K = self.kernel(X, sigma)
        
        # Compute unnormalized graph Laplacian
        L = self.laplacian(K)
        
        # Find k smallest eigenvalues with corresponding k eigenvectors of L
        val, vec = scipy.sparse.linalg.eigs(L, which='SR')
        val = val.real
        vec = vec.real[:, val.argsort()]
        val = np.sort(val)
        self.eigval = val
        self.eigvec = vec.T
        Z = vec[:,:k].T
        Y = Z / (np.linalg.norm(Z) ** (1/2))
        Y = Y.T
        
        # Apply k-means clustering
        kmeans = mykmeans(k)
        kmeans.fit(Y)
        self.labels = deepcopy(kmeans.labels)
        self.centers = deepcopy(kmeans.centers)
        self.num_iter = deepcopy(kmeans.num_iter)
