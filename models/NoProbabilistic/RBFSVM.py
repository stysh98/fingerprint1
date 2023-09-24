import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def dual_objective(K):
    def obj_svm(alpha):
        alpha = alpha.reshape((alpha.size, 1))
        gradient = (K.dot(alpha) - np.ones((alpha.shape[0], 1))).reshape((1, alpha.size))
        obj_l = 0.5 * alpha.T.dot(K).dot(alpha) - alpha.T @ np.ones(alpha.shape[0])
        return obj_l, gradient

    return obj_svm

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)    

def train_rbf_svm(X, y, pi, C, gamma=0.1):
    y = y.astype('int64')
    num_samples, num_features = X.shape
    classes = np.unique(y)
    num_classes = len(classes)
    
    # Calculate class priors and empirical priors
    class_counts = np.bincount(y)
    empirical_priors = class_counts / num_samples
    
    alphas = np.zeros((num_samples, num_classes))
    
    y_i = np.where(y == 1, 1, -1)
    bounds = np.zeros_like(y)
    bounds[y==0] = C * (pi / empirical_priors[classes[0]])
    bounds[y==1] = C * (pi / empirical_priors[classes[1]])
    
    ghat = np.array([[rbf_kernel(X[j], X[k], gamma=gamma) for k in range(num_samples)] for j in range(num_samples)])
    K = np.outer(y_i, y_i) * ghat
    
    initial_alpha = np.zeros(num_samples)
    bounds = list(zip(np.zeros_like(y), bounds))
    
    result = fmin_l_bfgs_b(dual_objective(K), x0=initial_alpha, bounds=bounds)
    alphas = result[0]
    
    return alphas
    
def predict_rbf_svm(X, alphas, x_train, y_train, gamma=0.1):
    dist = np.zeros((x_train.shape[0], X.shape[0]))
    y_i = np.where(y_train == 1, 1, -1)
    for i in range(x_train.shape[1]):
        for j in range(X.shape[0]):
            dist[i][j] += np.exp(-gamma * np.linalg.norm(x_train[i:i+1, :] - X[j:j+1, :]) ** 2)
    m = alphas.reshape((alphas.size, 1)) * y_i.reshape((y_i.size, 1))
    predictions = (m * dist).sum(0)
    return predictions

