import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV

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

clf = LogisticRegressionCV(multi_class='ovr')
clf.fit(x_train, y_train)

# Validation set
score = clf.score(x_val, y_val)
print('Multi-class classification score on validation set is %.2f' % score)

# Test set
pred = clf.predict(x_test)

# Write to submission.csv
df = pd.read_csv('sample_submission.csv')
df['Category'] = pred
df.to_csv('submission.csv',index=False)
