import numpy as np

class MultivariateGaussianTiedClassifier:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X_train, y_train):
        self.mean_vectors = []
        self.tied_cov_matrix = None
        self.tied_cov_matrix = np.cov(X_train, rowvar=False)
        
        for class_label in range(self.num_classes):
            class_samples = X_train[y_train == class_label]
            class_mean = np.mean(class_samples, axis=0)
            self.mean_vectors.append(class_mean)

    def predict(self, X_test):
        scores = []
        for sample in X_test:
            class_probs = []
            for class_label in range(self.num_classes):
                mean = self.mean_vectors[class_label]
                cov = self.tied_cov_matrix

                const = - 0.5 * len(sample) * np.log(2*np.pi)
                logdet = np.linalg.slogdet(cov)[1]
                L = np.linalg.inv(cov)
                v = np.dot((sample - mean).T, np.dot(L, ((sample - mean)))).ravel()
                pdf = const - 0.5 * logdet - 0.5 * v
                class_probs.append(pdf[0])

            scores.append(class_probs)
        scores = np.array(scores)
        scores = scores[:,1] - scores[:,0]
        return scores