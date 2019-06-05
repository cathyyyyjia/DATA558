import numpy as np
from sklearn.preprocessing import StandardScaler

def stdData(X):
    """
    Standardize the data
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(X)
    return x

def subsetData(X, Y, class1, class2, ovo=True):
    """
    Subset data
    """
    x = X
    y = Y
    if ovo == True:
        idx = np.concatenate((np.where(Y==class1),np.where(Y==class2)), axis=1).reshape(-1)
        x = x[idx]
        y = y[idx]
        # Standardize
        x = stdData(x)
        # Change label to +/- 1
        y[y==class1] = -1
        y[y==class2] = 1
    else:
        # Standardize
        x = stdData(x)
        # Change label to +/- 1
        y[y==class1] = 1
        y[y!=class1] = -1
    
    return x, y