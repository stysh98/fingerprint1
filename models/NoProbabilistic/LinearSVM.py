import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def dual_objective(K):
    def obj_svm(alpha):
        alpha = alpha.reshape((alpha.size, 1))
        gradient = (K.dot(alpha) - np.ones((alpha.shape[0], 1))).reshape((1, alpha.size))
        obj_l = 0.5 * alpha.T.dot(K).dot(alpha) - alpha.T @ np.ones(alpha.shape[0])
        return obj_l, gradient

    return obj_svm

def train_linear_svm(X, y, pi, C):
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
    dhat = np.hstack([X, np.ones((X.shape[0],1))])
    ghat = np.dot(dhat, dhat.T)
    
    K = np.outer(y_i, y_i) * ghat
    
    initial_alpha = np.zeros(num_samples)
    bounds = list(zip(np.zeros_like(y), bounds))
    
    result = fmin_l_bfgs_b(dual_objective(K), x0=initial_alpha, bounds=bounds)
    alphas = result[0]
    w = np.dot(dhat.T, alphas.reshape((alphas.size, 1)) * y_i.reshape((y_i.size, 1)))
    
    return w
    
def predict_linear_svm(X, w):
    d = np.hstack([X, np.ones((X.shape[0],1))])
    predictions = np.dot(d, w).ravel()
    return predictions

