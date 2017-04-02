from fastdtw import fastdtw
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



TrainingRealPCA = np.load("TrainingRealPCA.npy")
TrainingSpoofPCA = np.load("TrainingSpoofPCA.npy")
np.savetxt('TrainingRealPCA.txt',TrainingRealPCA,delimiter = ',')
np.savetxt('TrainingSpoofPCA.txt',TrainingSpoofPCA,delimiter=',')
y = np.ndarray(120)
TrainingRealPCA = TrainingRealPCA.astype(float)/np.max(TrainingRealPCA)
#TrainingSpoofPCA = TrainingSpoofPCA.astype(float)/np.max(TrainingSpoofPCA)
for i in range(0,120):
    y[i] = np.sum(TrainingRealPCA[i,:])/120

print y
lda = LinearDiscriminantAnalysis(n_components=120)
X_r2 = lda.fit(X = TrainingRealPCA.asmatrix(), y = y.asmatrix())
print np.asarray(X_r2)
