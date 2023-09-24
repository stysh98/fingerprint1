import numpy as np
from data.pca import *
from data.znorm import *
from metrics.minDCF import *
from termcolor import colored
from prettytable import PrettyTable 
from data.analyze_dataset import read_data
from models.Discriminative.LogisticRegression import *

def vec_xxT(x):
    x = x.reshape((x.size, 1))
    return np.dot(x, x.T).reshape(x.size ** 2)


def train_discriminative(file_path, file_path1, Quadratic = False):
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
    save_file_path = 'results/eval results/' + 'discriminative' + '.txt'
    pca_array = [None, 7]
    z_nrom_array = [False, True]
    pi_list = [0.1]
    l_list = []
    for i in range(-5,3):
        l_list.append(10**i)
    open(save_file_path, 'w').close()

    x_table = PrettyTable()
    x_table.field_names = ["PCA", "pi", "lambda", "z-norm", "minDCF"]
    model_name = 'model: Logistic Regression with Quadratic = ' + str(Quadratic)
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
                for l in l_list:
                    score_array = np.array([])
                    y_array = np.array([])

                    if Quadratic == True:
                        X_tr = np.apply_along_axis(vec_xxT, 1, X_p)
                        X_p = np.hstack([X_tr, X_p])
                        X_te = np.apply_along_axis(vec_xxT, 1, Xt_p)
                        Xt_p = np.hstack([X_te, Xt_p])

                    classifier = LogisticRegressionClassifier(l, pi)
                    classifier.fit(X_p, y)
                    predictions = classifier.predict(Xt_p)
                    score_array = np.hstack((score_array,predictions))
                    y_array = np.hstack((y_array,y_test))

                    minDCF_value = minDCF(y_array, score_array, pi, c_fn, c_fp)
                    x_table.add_row([pca_item, pi, l, z_norm_item, minDCF_value])
                    items.append(minDCF_value)
                    print("lambda", l, "Finished")
                print(colored('all lambda tested for ' + str(pi), 'green'))
                plt.plot(l_list, items)
                plt.xscale('log')
                plt.xticks(l_list);
                legends.append('pca ' + str(pca_item) + ' Znorm ' + str(z_norm_item))
        plt.legend(legends)
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        plt.title(pi)
        plt.savefig('results/Logistic Regression figures/'+ 'Test_LR_' + str(int(pi*10))+ '_Quadratic_' + str(Quadratic))
        plt.show()
    with open(save_file_path, 'a', encoding="utf-8") as f:
        f.write(model_name + "\n" + x_table.get_string() + "\n\n")
    print(x_table,"\n")