import cv2
import numpy as np
im=cv2.imread("001_L_1.png")
im_bw= cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)[1]
cv2.imshow('im_bw', im_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
x=np.asarray(im)
print x[100,23,0]
print x[100,23,1]
print x[100,23,2]
