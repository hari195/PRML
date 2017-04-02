import numpy as np
a = np.load("TestPCA.npy")


np.savetxt('TestPCA.txt',a,delimiter = ',')
