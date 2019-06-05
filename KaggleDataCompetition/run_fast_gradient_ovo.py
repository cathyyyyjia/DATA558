import numpy as np
import pandas as pd
from itertools import combinations
from prepare import stdData, subsetData
from fast_gradient import initEta, fastgradalgo, multiPredict

# Load data
x_train = np.load('train_features.npy')
y_train = np.load('train_labels.npy')
x_val = np.load('val_features.npy')
y_val = np.load('val_labels.npy')
x_test = np.load('test_features.npy')

print('Training set')
print('Number of observations: %d' % x_train.shape[0])
print('Number of dimension: %d' % x_train.shape[1])
print('\nValidation set')
print('Number of observations: %d' % x_val.shape[0])
print('Number of dimension: %d' % x_val.shape[1])

# Train OvO classifiers
arr = [i for i in range(100)]
pairs = list(combinations(arr, 2))
clfs = []

n, d = x_train.shape
beta_theta_init = np.zeros([4096,])
lamb = 1

for pair in pairs:
    class1 = pair[0]
    class2 = pair[1]
    xtrain, ytrain = subsetData(x_train, y_train, class1, class2, ovo=True)
    eta = initEta(xtrain, lamb)
    vals_fg = fastgradalgo(beta_theta_init, eta_init, X, Y, lamb)
    clf = vals_fg[-1:].reshape(-1)
    clfs.append(clf)


# Validation set
pred = multiPredict(x_val, pairs, clfs, ovo=True)
score = np.mean(pred==y_val)
print('Multi-class classification score on validation set is %.2f' % score)

# Test set
pred = multiPredict(x_test, pairs, clfs, ovo=True)

# Write to submission.csv
df = pd.read_csv('sample_submission.csv')
df['Category'] = pred
df.to_csv('submission.csv',index=False)
