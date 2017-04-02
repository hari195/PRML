import numpy as np
import matplotlib.pyplot as plt
import math

#Fucntion to calculate pdf
def gaussianND(X, mu, sigma):
    n = X.shape[1]
    meanDiff = np.subtract(X, mu)

    num = np.exp(-0.5 * np.array([np.sum((np.dot(np.dot(meanDiff,np.linalg.inv(sigma)),meanDiff.T)),axis=1)]))
    pdf = float(1)*num /math.sqrt(np.abs(math.pow(2*np.pi,n)*np.linalg.det(sigma)))

    return pdf


def weightedAverage(weights, values):
    weights = np.array([weights])
    val = np.dot(weights , values)
    val = val / np.sum(weights)
    return val

#Feature extraction
def pca(X):
    cov_mat = np.cov(X)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    matrix_w = np.hstack((eig_pairs[0][1]))
    transformed = matrix_w.T.dot(X)
    return transformed


X = []
#Importing required data here
TrainingReal = np.load("Test.npy")
TrainingSpoof = np.load("TrainingSpoof.npy")

#Performing Feature extraction
for i in range(0,400):
    X.append(pca(TrainingReal[i].T))

#resulting 1D arrays are stacked here
X = np.asarray(X).T
print X.shape
X = TrainingReal[:,:,0]
print X.shape
np.save("TestPCA.npy",X)
X = X.astype(float)/np.max(X)

#No. of data points
m = X.shape[0]

#No. of gaussian in mixture
k = 2
#Data vector length
n = X.shape[1]

#Setting a random mu
indices = np.random.permutation(m)
mu = X[indices[0:k],0]


sigma = []

#Setting covariance of complete data as initial
for j in range(0,k):
    sigma.append(np.cov(X))


#Weights initialized with equal values
phi = np.ones((1,k)) * (float(1)/k)

W = np.zeros((m,k))

for iter in range(0,1000):

    flag = 0
    print ' EM iteration ', iter, '\n'

    pdf = np.zeros((m,k))


    for j in range(0,k):
        pdf[:,j] = gaussianND(X, mu[j], sigma[j])

    pdf_w = np.multiply(pdf, phi)

    W = np.divide(pdf_w.astype(float), np.array([np.sum(pdf_w,axis=1)]).T)

    prevMu = np.copy(mu)

    for j in range(0,k):
        #Update of parameters
        phi[0][j] = np.mean(W[:,j], axis=0)
        mu[j, :] = weightedAverage(W[:,j], X)
        sigma_k = np.zeros((n,n))
        Xm = np.subtract(X,mu[j,:])

        for i in range(0,m):
            sigma_k = sigma_k + W[i,j]*np.dot(np.array([Xm[i,:]]).T, np.array([Xm[i,:]]))
        sigma[j] = np.divide(sigma_k, sum(W[:,j]))

    if np.array_equal(mu, prevMu):
        break
print mu, '\n\n', sigma, '\n\n'
