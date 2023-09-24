import numpy as np
from data.pca import *
from data.znorm import *
from metrics.minDCF import *
from termcolor import colored
from prettytable import PrettyTable 
from data.analyze_dataset import read_data
from models.Gaussian.NaiveBayesGaussian import *
from models.Gaussian.MultivariateGaussian import *
from models.Gaussian.MultivariateGaussianTied import *


def train_gaussian(file_path):
    dataset = read_data(file_path)
    print('Data reading is finished')
    print('='*18, 'training', '='*18)

    ########## use K-fold (k=5) ##########
    # Given parameters
    X = dataset[:,:-1]
    y = dataset[:,-1]
    k = 5
    N = int(dataset.shape[0]/k)
    pi = 0.5  # Prior probability of target class
    c_fn = 1  # Cost of false negative
    c_fp = 10  # Cost of false positive
    models = {"Multivariate Gaussian Classifier": MultivariateGaussianClassifier(2),
            "MVG with Tied Covariances": MultivariateGaussianTiedClassifier(2),
            "Naive Bayes Gaussian Classifier": NaiveBayesGaussianClassifier(2)
            }
    save_file_path = 'results/model results/' + 'gaussian' + '.txt'
    pca_array = [None, 9, 8, 7]
    z_nrom_array = [False, True]
    open(save_file_path, 'w').close()
    np.random.seed(0)
    indexes = np.random.permutation(dataset.shape[0])    
    for model in models:
        x_table = PrettyTable()
        x_table.field_names = ["PCA", "z-norm", "minDCF(π = 0.5)"]
        model_name = 'model: ' + model
        print(colored(model_name, 'light_yellow'))
        for z_norm_item in z_nrom_array:
            if z_norm_item:
                X_z = z_norm(X)
            else:
                X_z = X
            for pca_item in pca_array:
                if pca_item:
                    X_p = perform_pca(X_z, pca_item)
                else:
                    X_p = X_z
                score_array = np.array([])
                y_array = np.array([])
                for i in range(k):
                    idxTest = indexes[i*N:(i+1)*N]
        
                    if i > 0:
                        idxTrainLeft = indexes[0:i*N]
                    elif (i+1) < k:
                        idxTrainRight = indexes[(i+1)*N:]
                
                    if i == 0:
                        idxTrain = idxTrainRight
                    elif i == k-1:
                        idxTrain = idxTrainLeft
                    else:
                        idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
                    
                    X_train = X_p[idxTrain, :]
                    y_train = y[idxTrain]
                    X_val = X_p[idxTest, :]
                    y_val = y[idxTest]

                    classifier = models[model]
                    classifier.fit(X_train, y_train)
                    predictions = classifier.predict(X_val)
                    score_array = np.hstack((score_array,predictions))
                    y_array = np.hstack((y_array,y_val))

                minDCF_value = minDCF(y_array, score_array, pi, c_fn, c_fp)
                x_table.add_row([pca_item, z_norm_item, minDCF_value])
        with open(save_file_path, 'a', encoding="utf-8") as f:
            f.write(model_name + "\n" + x_table.get_string() + "\n\n")
        print(x_table,"\n")