import numpy as np
import sklearn.metrics
from scipy.stats import mode

def kernel_linear(X, Y=None):
    '''
    Compute linear kernel
    '''
    if Y is None:
        Y = X
    return X.dot(Y.T)

def obj(alpha, K, y, lamb, h=0.5):
    '''
    Find objective value
    '''
    n, d = K.shape
    loss = 0
    for i in range(n):
        diff = 1 - y[i] * (K.dot(alpha)[i])
        if diff < h:
            loss += 0
        elif diff >  h:
            loss += diff
        else:
            loss += ((diff + h)**2) / (4*h)
    obj = loss / n + lamb*alpha.T.dot(K).dot(alpha)
    return obj

def computegrad(alpha, K, y, lamb, h=0.5):
    '''
    Find gradient of F
    '''
    n, d = K.shape
    loss = 0
    for i in range(n):
        diff = 1 - y[i] * (K.dot(alpha)[i])
        if diff < h:
            loss += 0
        elif diff >  h:
            loss -= y[i] * K[i]
        else:
            loss -= (y[i] * K[i] * (diff + h)) / (2*h)
    grad = loss / n + 2 * lamb * (K.dot(alpha))
    return grad

def backtracking(beta, grad, K, y, lamb, eta=1, max_iter=20):
    '''
    Apply backtracking rule to find eta
    '''
    norm_grad = np.linalg.norm(grad)
    found_eta = 0
    t = 0
    while t < max_iter:
        if obj(beta-eta*grad, K, y, lamb) < obj(beta, K, y, lamb) - 0.5 * eta * norm_grad ** 2:
            break
        else:
            eta *= 0.8
            t += 1
    return eta

def initEta(X, lamb):
    """
    Get initialized eta value
    """
    n = X.shape[0]
    return 1/(max(np.linalg.eigvals(1/n*X.dot(X.T))) + lamb)

def mysvm(beta_theta_init, eta_init, K, y, lamb, max_iter=10, eps=0.001):
    '''
    Apply fast gradient descent to find beta values
    '''
    beta = beta_theta_init
    theta = beta_theta_init
    eta = eta_init
    grad = computegrad(theta, K, y, lamb)
    beta_vals = [beta]
    t = 0
    while np.linalg.norm(grad) > eps and t < max_iter:
        # backtracking rule
        eta = backtracking(theta, grad, K, y, lamb, eta=eta)
        beta_new = theta - eta * grad
        theta = beta_new + t/(t+3) * (beta_new - beta)
        grad = computegrad(theta, K, y, lamb)
        beta = beta_new
        beta_vals.append(beta_new)
        t += 1
    return np.array(beta_vals)

def predict(X, clf, class1, clasass2):
    """
    Make prediction with an individual classifier
    """
    n = len(X)
    pred = np.zeros(n)
    vals = np.zeros(n)
    for i in range(n):
        K = kerneleval_linear(X, X[i,:].reshape(1, -1)).reshape(-1)
        vals[i] = K.dot(clf)
    pred = np.sign(vals)
    pred = pred.astype(int)
    pred[pred==-1] = class1
    pred[pred==1] = class2
    return pred

def multiPredict(X, pairs, clfs, ovo=True):
    """
    Make prediction for multi class
    """
    preds = np.array([[0]]*X.shape[0])
    for i in range(len(clfs)):
        pred = predict(X, clfs[i], pairs[i][0], pairs[i][1])
        pred = np.array([pred]).T
        preds = np.concatenate((preds, pred), axis=1)
    if ovo:
        pred_final, cnt = mode(preds, axis=1)
    else:
        pred_final = max(preds, axis=1)
    return pred_final.reshape(-1)