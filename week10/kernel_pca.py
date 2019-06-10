import numpy as np
import scipy
from sklearn.cluster import SpectralClustering

class kernelPCA(object):
    
    def __init__(self, k, sigma=1, mode='kernel'):
        self.k = k
        self.sigma = sigma
        self.mode = mode

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
    
    def get_mode(self):
        """
        Return computation mode
        """
        return self.mode
    
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
        Kernel principal component analysis algorithm
        Input:
            - X: (n,d) array
        """
        k = self.get_k()
        sigma = self.get_sigma()
        mode = self.get_mode()
        
        # Compute RBF kernel
        K = self.kernel(X, sigma)
        
        if mode == 'kernel':
            # Find eigenvalues with corresponding eigenvectors of L
            val, vec = scipy.sparse.linalg.eigs(K, which='LR')
            val = val.real
            vec = vec.real
            self.eigval = val
            self.eigvec = vec.T
        
        elif mode == 'knn':
            # Find eigenvalues with corresponding eigenvectors of L
            val, vec = scipy.sparse.linalg.eigs(K, which='LR')
            val = val.real
            vec = vec.real
            self.eigval = val
            self.eigvec = vec.T
            Z = vec[:,:k].T
            Y = Z / (np.linalg.norm(Z) ** (1/2))
            Y = Y.T
            cluster = SpectralClustering(n_neighbors=k)
            cluster.fit(Y)
            self.labels = cluster.labels_
        
        elif mode == 'laplacian':
            # Compute unnormalized graph Laplacian
            L = self.laplacian(K)
            # Find k smallest eigenvalues with corresponding k eigenvectors of L
            val, vec = scipy.sparse.linalg.eigs(L, which='SR')
            val = val.real
            vec = vec.real[:, val.argsort()]
            val = np.sort(val)
            self.eigval = val
            self.eigvec = vec.T