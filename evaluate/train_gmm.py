import numpy as np
from data.pca import *
from data.znorm import *
from metrics.minDCF import *
from termcolor import colored
from prettytable import PrettyTable 
from data.analyze_dataset import read_data
from models.GMM.GMM import *


def train_gmm(file_path, file_path1):
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
    target_version = "tied"
    target_list = [0,1]
    nonTarget_version = "diagonal"
    nonTarget_list = [2, 3]
    save_file_path = 'results/eval results/' + 'gmm' + '.txt'
    pca_array = [None]
    z_nrom_array = [False, True]
    open(save_file_path, 'w').close()

    x_table = PrettyTable()
    x_table.field_names = ["PCA", "target", "nontarget", "z-norm", "minDCF"]
    for t in target_list:
        for nt in nonTarget_list:
            model_name = 'model: GMM'
            print(colored(model_name, 'light_yellow'))
            print(t,nt)
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
                    score_array = np.array([])
                    y_array = np.array([])
                    

                    DTR0 = X_p[y==0]                               
                    gmm_class0 = GMM_LBG(DTR0.T, nt, nonTarget_version)  
                    _, SM0 = logpdf_GMM(Xt_p.T, gmm_class0)                    
                
                    # same for class 1
                    DTR1 = X_p[y==1]                                  
                    gmm_class1 = GMM_LBG(DTR1.T, t, target_version)
                    _, SM1 = logpdf_GMM(Xt_p.T,gmm_class1)
                    
                    # compute scores
                    predictions = SM1 - SM0 

                    score_array = np.hstack((score_array,predictions))
                    y_array = np.hstack((y_array,y_test))

                    minDCF_value = minDCF(y_array, score_array, pi, c_fn, c_fp)
                    x_table.add_row([pca_item, t, nt, z_norm_item, minDCF_value])
    with open(save_file_path, 'a', encoding="utf-8") as f:
        f.write(model_name + "\n" + x_table.get_string() + "\n\n")
    print(x_table,"\n")