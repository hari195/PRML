import numpy as np
import cv2
import math


def find_val(w,mu,sigma,x):
    delta=x-mu
    deltat=np.transpose(delta)
    covariance=delta*deltat
    cinv=np.linalg.inv(covariance)
    m=w*math.exp(-0.5*delta*cinv*deltat)
    d=math.pow(math.sqrt(6.28),10)*math.pow(np.linalg.det(cinv),0.5)
    return m/d

# Import data from file
# Here I'm only importing 1 image
# This has to be repested for all 60 images and each image sould be stacked on top of each other
# Expected dimension 60x565x160
im=cv2.imread("001_L_1.png")
x=np.asarray(im)

# Initilaize 2 means, sigma, and weights
mu1 = np.ones((10,), dtype=np.int)
sigma1 = np.ones((10,), dtype=np.int)
w1 = 0.5

mu2 = np.ones((10,), dtype=np.int)
sigma2 = np.ones((10,), dtype=np.int)
w2 = 0.5

# Assign constants like no. of samples, no. of features.
n=60
k=2

for num in range(1,1000):
    # Compute initial Expectation with the given values, and assign each data point to either of the class
    for i in range(0,59):
        cluster1[i] = find_val(w1,mu1,sigma1,x[i])
        cluster2[i] = find_val(w2,mu2,sigma2,x[i])
        c1[i]=cluster1[i]/(cluster1[i]+cluster2[i])
        c2[i]=1-c1[i]

    prevmu1=mu1
    prevmu2=mu2

    # Recompute 6 paramters.
    w2=1-w1
    # Check for convergence or Repeat
    if prevmu1==mu1 and prevmu2==mu2:
        break
