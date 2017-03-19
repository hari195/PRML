import numpy as np
import cv2
import math
from numpy.linalg import inv
from numpy.linalg import det

def find_val(w,mu,sigma,x):
    delta=x-mu
    deltat=np.transpose(delta)
    covariance=np.cov(x)
    cinv=inv(covariance)
    m=w*np.exp(np.multiply(-0.5,np.dot(np.dot(deltat,cinv),delta)))
    d=math.sqrt(6.28*np.linalg.det(cinv))
    return m/d

# Import data from file
# Here I'm only importing 1 image
# This has to be repested for all 60 images and each image sould be stacked on top of each other
# Expected dimension 60x565x160
TrainingReal = np.load("TrainingReal.npy")
TrainingSpoof = np.load("TrainingSpoof.npy")


# Initilaize 2 means, sigma, and weights
mu1 = TrainingReal[1]
sigma1 = TrainingReal[1]
w1 = 0.5

mu2 = TrainingSpoof[1]
sigma2 = TrainingSpoof[1]
w2 = 0.5

# Assign constants like no. of samples, no. of features.
n=60
k=2

for num in range(1,1000):
    # Compute initial Expectation with the given values, and assign each data point to either of the class
    for i in range(0,119):
        cluster1[i] = find_val(w1,mu1,sigma1,TrainingReal[i])
        cluster2[i] = find_val(w2,mu2,sigma2,TrainingReal[i])
        c1[i]=cluster1[i]/(cluster1[i]+cluster2[i])
        c2[i]=1-c1[i]

    prevmu1=mu1
    prevmu2=mu2

    # Recompute 6 paramters.
    w2=1-w1
    # Check for convergence or Repeat
    if prevmu1==mu1 and prevmu2==mu2:
        break
