import numpy as np

def z_norm(data):
    # Calculate mean and standard deviation
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    # Calculate Z-scores
    z_scores = (data - mean) / std_dev
    return z_scores