import numpy as np
import matplotlib.pyplot as plt

def perform_pca(data, num_components):

    # Standardize the data (mean = 0, variance = 1)
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    standardized_data = (data - mean) / std_dev

    # Calculate the covariance matrix
    covariance_matrix = np.cov(standardized_data, rowvar=False)

    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in decreasing order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Choose the number of principal components to retain
    selected_eigenvectors = eigenvectors[:, :num_components]

    # Transform the data to the new PCA space
    pca_data = np.dot(standardized_data, selected_eigenvectors)

    return pca_data