import numpy as np
import cv2
import math
from numpy.linalg import inv
from numpy.linalg import det

def find_val(w,mu,cov,x):
    delta=x-mu
    deltat=np.transpose(delta)
    cinv=inv(cov)
    m=np.multiply(-0.5,np.matmul(np.dot(deltat,cinv),delta))
    print m
    m=w*(2.71828 ** m)
    d=math.sqrt((6.28)*np.linalg.det(cinv))

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

mu2 = 20
sigma2 = np.identity(565)
w2 = 0.5

# Assign constants like no. of samples, no. of features.
n=60
k=2
p1 = np.zeros(120,)
p2 = np.zeros(120,)
p1x = np.zeros((120,565), dtype= np.int)
p2x = np.zeros((120,565), dtype=np.int)
prevmu1 = 0
prevmu2 = 0
p1xx = np.zeros((120,565,565), dtype= np.int)
p2xx = np.zeros((120,565,565), dtype=np.int)


for num in range(1,1000):
    print "Iteration", num
    # Compute initial Expectation with the given values, and assign each data point to either of the class
    for i in range(0,120):
        x = TrainingReal[i][0]

        a = find_val(w1,mu1,sigma1,x)

        b = find_val(w2,mu2,sigma2,x)

        p1[i]=a/(a+b)
        p2[i]=b/(a+b)
        p1x[i]=np.multiply(x,p1[i])
        p2x[i]=np.multiply(x,p2[i])

        delta=x-mu1
        delta = delta.reshape((-1,1))
        deltat = delta.reshape((-1,1)).reshape(1,565)

        p1xx[i] = np.multiply(p1[i],np.dot(delta,deltat))

        delta=x-mu2
        delta = delta.reshape((-1,1))
        deltat = delta.reshape((-1,1)).reshape(1,565)
        p2xx[i] = np.multiply(p2[i],np.dot(delta,deltat))


    prevmu1=mu1
    prevmu2=mu2

    # Recompute 6 paramters.
    w1=np.sum(p1)/120
    w2=1-w1
    mu1=np.sum(p1x)/np.sum(p1)
    mu2=np.sum(p2x)/np.sum(p2)

    sigma1 = np.sum(p1xx)/np.sum(p1)
    sigma2 = np.sum(p2xx)/np.sum(p2)
    print np.sum(p1)
    print np.sum(p1xx)

    # Check for convergence or Repeat
    if prevmu1==mu1 and prevmu2==mu2:
        break
