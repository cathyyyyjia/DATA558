import numpy as np
import pandas as pd
from itertools import combinations
from prepare import stdData, subsetData
from svm import initEta, kerneleval_linear, mysvm, multiPredict

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

# Train OvR classifiers
arr = [i for i in range(100)]
pairs = []
clfs = []

# Train SVM with linear kernel
std_x_train = stdData(x_train)
K = kernel_linear(std_x_train)
lamb = 1
eta = initEta(K, lamb)

for cls in arr:
    pairs.append([cls, -1])
    xtrain, ytrain = subsetData(x_train, y_train, cls, -1, ovo=False)
    n, d = xtrain.shape
    betas = mysvm(np.zeros(n), eta, K, ytrain, lamb)
    clf = betas[-1:].reshape(-1)
    clfs.append(clf)

# Validation set
pred = multiPredict(x_val, pairs, clfs, ovo=False)
score = np.mean(pred==y_val)
print('Multi-class classification score on validation set is %.2f' % score)

# Test set
pred = multiPredict(x_test, pairs, clfs, ovo=False)

# Write to submission.csv
df = pd.read_csv('sample_submission.csv')
df['Category'] = pred
df.to_csv('submission.csv',index=False)
