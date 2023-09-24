from data.analyze_dataset import *

file_path = "data/Train.txt"
print('*'*18, 'read data', '*'*18)
dataset = read_data(file_path)
print('*'*18, 'plot features', '*'*18)
plot_features(dataset, ['spoofed fingerprint', 'authentic fingerprint'])
print('*'*18, 'LDA', '*'*18)
LDA(dataset, ['spoofed fingerprint', 'authentic fingerprint'])
print('*'*18, 'pca2', '*'*18)
plot_pca2(dataset, ['spoofed fingerprint', 'authentic fingerprint'])
print('*'*18, 'Correlation Heatmaps', '*'*18)
# define data and spoofed fingerprint and authentic fingerprint for creating heatmap
X = dataset[:,:-1]
y = dataset[:,-1]
condition1 = np.where(y == 0)
condition2 = np.where(y == 1)
data1 = X[condition1]
data2 = X[condition2]

feature_labels = np.arange(1,X.shape[1]+1)
correlation_heatmap(X, labels=feature_labels, title="Feature Correlation Heatmap", cmap="gray_r", name= 'data')
correlation_heatmap(data1, labels=feature_labels, title="Feature Correlation Heatmap", cmap="Blues", name='spoofed fingerprint')
correlation_heatmap(data2, labels=feature_labels, title="Feature Correlation Heatmap", cmap="Reds", name= 'authentic fingerprint')
cross_features(dataset, 3, 0, ['spoofed fingerprint', 'authentic fingerprint'])
cross_features(dataset, 5, 6, ['spoofed fingerprint', 'authentic fingerprint'])
cross_features(dataset, 0, 7, ['spoofed fingerprint', 'authentic fingerprint'])
cross_features(dataset, 6, 8, ['spoofed fingerprint', 'authentic fingerprint'])