

n = 151        #no. of training examples
d = 1          #no. of features
c = 1                          #no. of classes
nc = 151                        #no. of training examples in each class
ng = 5                #no. gaussians in GMM for each class
m = 0
it = 10
import numpy as np
import math
from numpy import ndarray

#Feature extraction
def pca(X):
    cov_mat = np.cov(X)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    matrix_w = np.hstack((eig_pairs[0][1]))
    transformed = matrix_w.T.dot(X)
    return transformed


def gaussian(u,C,x,d):
    #C = np.diag(np.diag(C))
    C_det = np.linalg.det(C)
    #print C_det
    C_inv = np.linalg.inv(C)

    t4 = np.matrix(x-u)
    t5 = np.matrix(C_inv)
    if C_det == 0:
        C_det = 1
    t7 = math.sqrt(pow(2*math.pi, d) * C_det)

    t8 = np.dot(np.dot(t4, t5), t4.T)
    t9 = (1.0/t7) * math.exp(-t8/2)

    return t9

weights1 =  ndarray((ng,c), np.float64)
means1 =  ndarray((d,ng,c), np.float64)
Cov_Matrices1 =  ndarray((d,d,ng,c), np.float64)

x = np.load("TrainingRealLDA2.npy")

#INITIALISING GAUSSIANS
u = np.random.randn(d,ng)
Cmat = ndarray((d,d,ng), np.float64)
for k in range(0,ng):
    temp1 = np.random.randn(d)
    temp2 = np.exp(temp1)
    Cmat[:,:,k] = np.diag(temp2)
    alpha = np.ones(shape=(ng))/ng

#TRAINING THE GMM
for iter in range(0,it):

    #FINDING POSTERIOR PROBABILITIES FOR EACH TRAINING EXAMPLE BELONGING TO EACH GAUSSIAN FUNCITON
    temp1 = np.zeros(shape=(nc,ng))
    for k in range(0, ng):
        uk = np.matrix(u[:, k])
        Ck = np.matrix(Cmat[:, :, k])
        for i in range(0,nc):
            xi = np.matrix(x[i])
            #print Ck
            temp1[i,k] = gaussian(uk,Ck,xi,d)*alpha[k]
    temp2 = np.sum(temp1, axis=1)
    temp3 = np.divide(temp1.T,temp2.T)
    p_xi_in_k = np.matrix(temp3.T)

    #RECOMPUTING THE PARAMETERS FOR EACH GAUSSIAN
    for k in range(0, ng):
        temp_alphak = np.mean(p_xi_in_k[:,k])

        temp_uk = (np.dot(x.T, p_xi_in_k[:,k])) / (temp_alphak*nc)
        temp_uk = np.array(temp_uk)

        temp = np.zeros(shape=(d,d))
        uk = u[:,k]
        for i in range(0,nc):
            xi = np.matrix(x[i])

            temp = temp + (np.outer(xi-uk, xi-uk) * p_xi_in_k[i,k])
        temp_Ck = temp/ (temp_alphak*nc)
        #print temp_Ck
        alpha[k] = temp_alphak
        u[:,k] = temp_uk[:,0]
        Cmat[:,:,k] = temp_Ck

weights1[:,m] = alpha
means1[:,:,m] = u
Cov_Matrices1[:,:,:,m] = Cmat
print(weights1)
print(weights1.shape)
print('--------------------')
print(means1)
print(means1.shape)
print('--------------------')
print(Cov_Matrices1)
print(Cov_Matrices1.shape)

print('FIRST CLASS DONE')

n = 151        #no. of training examples
d = 1          #no. of features
c = 1                          #no. of classes
nc = 151                        #no. of training examples in each class
ng = 5                #no. gaussians in GMM for each class
m = 0

def gaussian(u,C,x,d):
    C = np.diag(np.diag(C))
    C_det = np.linalg.det(C)
    #print C_det
    C_inv = np.linalg.inv(C)

    t4 = np.matrix(x-u)
    t5 = np.matrix(C_inv)
    if C_det == 0:
        C_det = 1
    t7 = math.sqrt(pow(2*math.pi, d) * C_det)

    t8 = np.dot(np.dot(t4, t5), t4.T)
    t9 = (1.0/t7) * math.exp(-t8/2)

    return t9

weights2 =  ndarray((ng,c), np.float64)
means2 =  ndarray((d,ng,c), np.float64)
Cov_Matrices2 =  ndarray((d,d,ng,c), np.float64)

x = np.load("TrainingSpoofLDA2.npy")


#INITIALISING GAUSSIANS
u = np.random.randn(d,ng)
Cmat = ndarray((d,d,ng), np.float64)
for k in range(0,ng):
    temp1 = np.random.randn(d)
    temp2 = np.exp(temp1)
    Cmat[:,:,k] = np.diag(temp2)
alpha = np.ones(shape=(ng))/ng

#TRAINING THE GMM
for iter in range(0,it):

    #FINDING POSTERIOR PROBABILITIES FOR EACH TRAINING EXAMPLE BELONGING TO EACH GAUSSIAN FUNCITON
    temp1 = np.zeros(shape=(nc,ng))
    for k in range(0, ng):
        uk = np.matrix(u[:, k])
        Ck = np.matrix(Cmat[:, :, k])
        for i in range(0,nc):
            xi = np.matrix(x[i])
            temp1[i,k] = gaussian(uk,Ck,xi,d)*alpha[k]
    temp2 = np.sum(temp1, axis=1)
    temp3 = np.divide(temp1.T,temp2.T)
    p_xi_in_k = np.matrix(temp3.T)

    #RECOMPUTING THE PARAMETERS FOR EACH GAUSSIAN
    for k in range(0, ng):
        temp_alphak = np.mean(p_xi_in_k[:,k])

        temp_uk = (np.dot(x.T, p_xi_in_k[:,k])) / (temp_alphak*nc)
        temp_uk = np.array(temp_uk)

        temp = np.zeros(shape=(d,d))
        uk = u[:,k]
        for i in range(0,nc):
            xi = np.matrix(x[i])
            temp = temp + (np.outer(xi-uk, xi-uk) * p_xi_in_k[i,k])
        temp_Ck = temp/ (temp_alphak*nc)

        alpha[k] = temp_alphak
        u[:,k] = temp_uk[:,0]
        Cmat[:,:,k] = temp_Ck

weights2[:,m] = alpha
means2[:,:,m] = u
Cov_Matrices2[:,:,:,m] = Cmat
print(weights2)
print(weights2.shape)
print('---------------------------')
print(means2)
print(means2.shape)
print('---------------------------')
print(Cov_Matrices2)
print(Cov_Matrices2.shape)

print('SECOND CLASS DONE-------------------------')

def gaussian2(u,C,x,d):

    #print C_det
    C_inv = 1/C
    t4 = x-u
    t7 = math.sqrt(pow(2*math.pi, 1) * C)
    t8 = (t4*t4)/(2*C*C)
    t9 = (1.0/t7) * math.exp(-t8)
    return t9

test = np.load("TestLDA.npy")


answers = [0,0,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,0,1,1,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,0]
answers = np.asarray(answers)
answers2 = np.ndarray(100)
prob = np.ndarray((5,2))
print answers
for sample in range(0,100):
    a = gaussian2(means1[:,1,:],Cov_Matrices1[:,:,1,:],test[sample],10) + gaussian2(means1[:,2,:],Cov_Matrices1[:,:,2,:],test[sample],10) + gaussian2(means1[:,3,:],Cov_Matrices1[:,:,3,:],test[sample],10) + gaussian2(means1[:,4,:],Cov_Matrices1[:,:,4,:],test[sample],10) + gaussian2(means1[:,0,:],Cov_Matrices1[:,:,0,:],test[sample],10)
    b = gaussian2(means2[:,1,:],Cov_Matrices2[:,:,1,:],test[sample],10) + gaussian2(means2[:,2,:],Cov_Matrices2[:,:,2,:],test[sample],10) + gaussian2(means2[:,3,:],Cov_Matrices2[:,:,3,:],test[sample],10) + gaussian2(means2[:,4,:],Cov_Matrices2[:,:,4,:],test[sample],10) + gaussian2(means2[:,0,:],Cov_Matrices2[:,:,0,:],test[sample],10)
    if a>b:
        answers2[sample] = int(1)
    else:
        answers2[sample]=int(0)
correct = 0.0
wrong = 0.0
answers2 = answers2.astype(int)
print answers2
for sample in range(0,100):
    if answers[sample]==answers2[sample]:
        correct = correct + 1
    else:
        wrong = wrong + 1


print "Accuracy:"
print correct/100
