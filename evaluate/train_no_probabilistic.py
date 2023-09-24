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

def train_SVM(file_path, file_path1):
    dataset = read_data(file_path)
    test_data = read_data(file_path1)
    print('Data reading is finished')
    print('='*18, 'training', '='*18)

    ########## use K-fold (k=5) ##########
    # Given parameters
    X = dataset[:,:-1]
    y = dataset[:,-1]
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1]
    
    pi = 0.5  # Prior probability of target class
    c_fn = 1  # Cost of false negative
    c_fp = 10  # Cost of false positive
    save_file_path = 'results/eval results/' + 'No_Probabilistic' + '.txt'
    pca_array = [None, 9, 8, 7]
    z_nrom_array = [False]
    pi_list = [0.1]
    C_list = []
    for i in range(-5,3):
        C_list.append(10**i)
    open(save_file_path, 'w').close()
  
    x_table = PrettyTable()
    x_table.field_names = ["PCA", "pi", "C", "z-norm", "minDCF"]
    model_name = 'model: RBF SVM'
    print(colored(model_name, 'light_yellow'))
    for pi in pi_list:
        legends = []
        for z_norm_item in z_nrom_array:
            if z_norm_item:
                X_z = z_norm(X)
                Xt_z = z_norm(X_test)
            else:
                X_z = X
                Xt_z = X_test
            for pca_item in pca_array:
                if pca_item:
                    X_p = perform_pca(X_z, pca_item)
                    Xt_p = perform_pca(Xt_z, pca_item)
                else:
                    X_p = X_z
                    Xt_p = Xt_z
                items = []
                for C in C_list:
                    score_array = np.array([])
                    y_array = np.array([])

                    alphas = train_polynomial_svm(X_p, y, pi, C)
                    predictions = predict_polynomial_svm(Xt_p, alphas, X_p, y)
                    score_array = np.hstack((score_array,predictions))
                    y_array = np.hstack((y_array,y_test))

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
        plt.savefig('results/SVM figures/'+ 'Test_R_SVM')
        plt.show()
    with open(save_file_path, 'a', encoding="utf-8") as f:
        f.write(model_name + "\n" + x_table.get_string() + "\n\n")
    print(x_table,"\n")