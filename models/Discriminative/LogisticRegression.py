import numpy as np
import scipy.optimize

class LogisticRegressionClassifier:
    def __init__(self, l, prior):
        self.l = l
        self.prior = prior
        self.w = None
        self.b = None

    def log_reg(self, X_train, y_train):
        def logreg_obj(v):
            w, b = v[:-1], v[-1]
            nt = X_train[y_train == 1].shape[0]
            nf = X_train[y_train == 0].shape[0]
            n = X_train.shape[0]
            st = 0
            sf = 0
            for i in range(n):
                xi = X_train[i, :]
                zi = (2*y_train[i]) - 1
                if(zi == 1):
                    st += np.logaddexp(0, -zi*(np.dot(w.T, xi) + b))
                elif(zi == -1):
                    sf += np.logaddexp(0, -zi*(np.dot(w.T, xi) + b))
            reg = self.l/2 * np.linalg.norm(w) ** 2
            return reg + (self.prior/nt) * st + ((1-self.prior)/nf) * sf
        return logreg_obj


    def fit(self, X_train, y_train):
        logreg_obj = self.log_reg(X_train, y_train)
        # Minimize the function func using the L-BFGS-B algorithm
        minimizer = scipy.optimize.fmin_l_bfgs_b(
            logreg_obj, np.zeros(X_train.shape[1] + 1), approx_grad=True)
        self.w, self.b = minimizer[0][:-1], minimizer[0][-1]

    def predict(self, X_test):
        score = np.dot(self.w,X_test.T) + self.b
        return score