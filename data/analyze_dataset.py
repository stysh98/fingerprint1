import numpy as np
import matplotlib.pyplot as plt
from data.pca import perform_pca

def read_data(file_path):
    '''
    read file path and return the dataset
    input(s):
      file_path (string): path of txt file
    output(s):
      dataset (ndarray): the dataset
    '''
    dataset = []
    # Open the file in read mode
    file = open(file_path, 'r')
    for f in file:
        # Split the content using commas
        split_content = f.split(',')
        split_content[-1] = split_content[-1][0]
        dataset.append(split_content)
    dataset = np.array(dataset).astype('float64')
    return dataset

def plot_features(dataset , label_names=[]):
    length = len(label_names)
    labels_column = dataset[:,-1]
    number_of_features = dataset.shape[1]
    labels = np.unique(labels_column).astype('int')
    if length != 0 and length !=len(labels):
        raise ValueError('the number of label names and features should be the same')
    if length == 0: label_names = labels
    for n in range(number_of_features-1):
        for l in labels:
            condition = np.where(labels_column == l)
            plt.hist(dataset[:, n][condition], density = True, bins=50, alpha=0.5, label=label_names[l], edgecolor='gray')
        plt.title('featre number' + str(n+1), alpha=0.7)
        plt.legend()
        plt.savefig('results/feature figures/'+ 'featre number' + str(n+1))
        plt.show()

def LDA(dataset, label_names=[]):
    X = dataset[:,:-1]
    y = dataset[:,-1]
    length = len(label_names)
    labels = np.unique(y).astype('int')
    if length != 0 and length !=len(labels):
        raise ValueError('the number of label names and features should be the same')
    if length == 0: label_names = labels
    condition1 = np.where(y == 0)
    condition2 = np.where(y == 1)
    data1 = X[condition1]
    data2 = X[condition2]
    # Compute class means
    mean_class1 = np.mean(data1, axis=0)
    mean_class2 = np.mean(data2, axis=0)
    # Compute the within-class scatter matrix
    scatter_within = np.dot((data1 - mean_class1).T, (data1 - mean_class1)) + \
                    np.dot((data2 - mean_class2).T, (data2 - mean_class2))
    # Compute the between-class scatter matrix
    scatter_between = np.outer(mean_class1 - mean_class2, mean_class1 - mean_class2)
    # Solve the generalized eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(scatter_within).dot(scatter_between))
    # Sort eigenvectors by eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    # Choose the top k eigenvectors (k is the desired dimensionality)
    k = 1
    transformation_matrix = eigenvectors[:, :k]
    # Project the data onto the new subspace
    lda_data = np.dot(X, transformation_matrix)
    # Plot the LDA-transformed data
    plt.hist(lda_data[y == 0], bins=50, alpha=0.5, label=label_names[0], edgecolor='gray')
    plt.hist(lda_data[y == 1], bins=50, alpha=0.5, label=label_names[1], edgecolor='gray')
    plt.xlabel('LDA Component')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of LDA-Transformed Data')
    plt.savefig('results/LDA figure/'+ 'LDA')
    plt.show()

def plot_pca2(dataset, label_names=[]):
    X = dataset[:,:-1]
    y = dataset[:,-1]
    length = len(label_names)
    labels = np.unique(y).astype('int')
    if length != 0 and length !=len(labels):
        raise ValueError('the number of label names and features should be the same')
    if length == 0: label_names = labels
    k = 2
    pca_data = perform_pca(X, k)
    # Plot the PCA-transformed data
    condition1 = np.where(y == 0)
    condition2 = np.where(y == 1)
    data1 = pca_data[condition1]
    data2 = pca_data[condition2]
    plt.scatter(data1[:,0], data1[:,1], alpha=0.7, label=label_names[0])
    plt.scatter(data2[:,0], data2[:,1], alpha=0.7, label=label_names[1])
    plt.xlabel('PCA Component')
    plt.ylabel('y')
    plt.title('PCA-Transformed Data')
    plt.savefig('results/pca2 figure/'+ 'pca2')
    plt.show()

def correlation_heatmap(hitmap_data, labels=None, title="Correlation Heatmap", cmap="gray_r", name='correlation heatmap'):
    correlation_matrix = np.corrcoef(hitmap_data, rowvar=False)
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap=cmap)
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(title)
    plt.savefig('results/correlation heatmaps/'+ name)
    plt.show()

def cross_features(dataset, n_feature1, n_feature2, label_names=[]):
    X = dataset[:,:-1]
    y = dataset[:,-1]
    length = len(label_names)
    labels = np.unique(y).astype('int')
    if length != 0 and length !=len(labels):
        raise ValueError('the number of label names and features should be the same')
    if length == 0: label_names = labels
    # Plot the cross features
    condition1 = np.where(y == 0)
    condition2 = np.where(y == 1)
    data1 = X[condition1]
    data2 = X[condition2]
    plt.scatter(data1[:,n_feature1], data1[:,n_feature2], alpha=0.7, label=label_names[0])
    plt.scatter(data2[:,n_feature1], data2[:,n_feature2], alpha=0.7, label=label_names[1])
    plt.xlabel('feature ' + str(n_feature1))
    plt.ylabel('feature ' + str(n_feature2))
    plt.title('cross features')
    plt.savefig('results/cross features/'+ str(n_feature1) + '_and_' + str(n_feature2))
    plt.show()