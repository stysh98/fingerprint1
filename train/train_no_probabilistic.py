import numpy as np
from data.pca import *
from data.znorm import *
from metrics.minDCF import *
from termcolor import colored
from prettytable import PrettyTable 
from data.analyze_dataset import read_data
from models.NoProbabilistic.LinearSVM import *
from models.NoProbabilistic.PolynomialSVM import *
from models.NoProbabilistic.RBFSVM import *

def train_SVM(file_path):
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
    save_file_path = 'results/model results/' + 'No_Probabilistic' + '.txt'
    pca_array = [None, 9, 8, 7]
    z_nrom_array = [False]
    pi_list = [0.1]
    C_list = []
    for i in range(-5,3):
        C_list.append(10**i)
    open(save_file_path, 'w').close()
    np.random.seed(0)
    indexes = np.random.permutation(dataset.shape[0])    
    x_table = PrettyTable()
    x_table.field_names = ["PCA", "pi", "C", "z-norm", "minDCF"]
    model_name = 'model: RBF SVM'
    print(colored(model_name, 'light_yellow'))
    for pi in pi_list:
        legends = []
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
                items = []
                for C in C_list:
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

                        alphas = train_rbf_svm(X_train, y_train, pi, C)
                        predictions = predict_rbf_svm(X_val, alphas, X_train, y_train)
                        score_array = np.hstack((score_array,predictions))
                        y_array = np.hstack((y_array,y_val))

                    minDCF_value = minDCF(y_array, score_array, pi, c_fn, c_fp)
                    x_table.add_row([pca_item, pi, C, z_norm_item, minDCF_value])
                    items.append(minDCF_value)
                    print("C", C, "Finished")
                print(colored('all C tested for ' + str(pi), 'green'))
                plt.plot(C_list, items)
                plt.xscale('log')
                plt.xticks(C_list);
                legends.append('pca ' + str(pca_item) + ' Znorm ' + str(z_norm_item))
        plt.legend(legends)
        plt.xlabel('C')
        plt.ylabel('minDCF')
        plt.title(pi)
        plt.savefig('results/SVM figures/'+ 'R_SVM')
        plt.show()
    with open(save_file_path, 'a', encoding="utf-8") as f:
        f.write(model_name + "\n" + x_table.get_string() + "\n\n")
    print(x_table,"\n")