import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

def obj(beta, X, Y, lamb):
    """
    Compute objective value
    """
    n = X.shape[0]
    return 1/n*np.sum(np.log(1+np.exp(np.multiply(-Y,X.dot(beta))))) + lamb*np.linalg.norm(beta)**2

def computegrad(beta, X, Y, lamb):
    """
    Compute gradient
    """
    n = X.shape[0]
    p = np.exp(np.multiply(-Y,X.dot(beta)))/(1+np.exp(np.multiply(-Y,X.dot(beta))))
    p = np.diag(p)
    return -1/n*X.T.dot(p).dot(Y.T)+2*lamb*beta

def backtracking(beta, grad, X, Y, lamb, eta, max_iter=20):
    """
    Apply backtracking rule to update eta value
    """
    norm_grad = np.linalg.norm(grad)
    t = 0
    while t < max_iter:
        if obj(beta-eta*grad, X, Y, lamb) < obj(beta, X, Y, lamb) - 0.5 * eta * norm_grad ** 2:
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

def fastgradalgo(beta_theta_init, eta_init, X, Y, lamb, max_iter=10, eps=0.001):
    """
    Apply fast gradient descent to find beta values
    """
    beta = beta_theta_init
    theta = beta_theta_init
    eta = eta_init
    grad = computegrad(theta, X, Y, lamb)
    beta_vals = [beta]
    t = 0
    # The stopping criterion is norm(grad) <= eps and maximum iteration limit
    while np.linalg.norm(grad) > eps and t < max_iter:
        eta = backtracking(theta, grad, X, Y, lamb, eta)
        beta_new = theta - eta * grad
        theta = beta_new + t/(t+3) * (beta_new - beta)
        grad = computegrad(theta, X, Y, lamb)
        beta = beta_new
        beta_vals.append(beta_new)
        t += 1
    return np.array(beta_vals)

def predict(X, beta, cls1, cls2, threshold=0.5):
    """
    Make prediction with an individual classifier
    """
    pred = 1/(1+np.exp(-X.dot(beta.T))) > threshold # logistic function
    pred = pred.astype(int) # True 1 False 0
    pred[pred==0] = cls1
    pred[pred==1] = cls2
    return pred.T

def compME(X, Y, beta, cls1, cls2):
    """
    Compute misclassification error
    """
    pred = predict(X, beta, cls1, cls2)
    err = np.mean(pred != Y)
    return err

def ME_plot(X1, Y1, X2, Y2, vals, cls1, cls2):
    """
    Make misclassification error plot
    """
    me1 = []
    me2 = []
    for val in vals:
        me1.append(compME(X1, Y1, val, cls1, cls2))
        me2.append(compME(X2, Y2, val, cls1, cls2))
    plt.figure()
    plt.plot(me1, label='Training Set')
    plt.plot(me2, label='Validation Set')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error', fontsize=13)
    plt.legend()
    plt.show()
    return

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