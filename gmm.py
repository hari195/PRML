import numpy as np
import cv2
import math
from numpy.linalg import inv
from numpy.linalg import det

def find_val(w,mu,cov,x):
    delta=x-mu
    deltat=np.transpose(delta)
    cinv=inv(cov)
    m=np.multiply(-0.5,np.dot(np.dot(deltat,cinv),delta))
    m=2.71828 ** m
    d=math.sqrt((6.28   )*np.linalg.det(cinv))
    print m
    return m/d

# Import data from file
# Here I'm only importing 1 image
# This has to be repested for all 60 images and each image sould be stacked on top of each other
# Expected dimension 120x150x565
TrainingReal = np.load("TrainingReal.npy")
TrainingSpoof = np.load("TrainingSpoof.npy")

# Initilaize 2 means, sigma, and weights
mu1 = 10
sigma1 = np.identity(565)
w1 = 0.5

mu2 = 10
sigma2 = np.identity(565)
w2 = 0.5

# Assign constants like no. of samples, no. of features.
n=60
k=2

for num in range(1,1000):
    # Compute initial Expectation with the given values, and assign each data point to either of the class
    for i in range(1,120):
        a =find_val(w1,mu1,sigma1,TrainingReal[i][0])
        b= find_val(w2,mu2,sigma2,TrainingReal[i][0])
        p1=a/(a+b)
        p2 = 1-p1
        


    prevmu1=mu1
    prevmu2=mu2

    # Recompute 6 paramters.
    w2=1-w1
    # Check for convergence or Repeat
    if prevmu1==mu1 and prevmu2==mu2:
        break
