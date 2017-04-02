
from tempfile import TemporaryFile
import numpy as np
from PIL import Image
import glob

outfile = TemporaryFile()
image_list = np.ones((400,150,565), dtype=np.int)
i=0
for filename in glob.glob('cropped/test/*.png'): #assuming gif
    im=Image.open(filename)
    x=np.asarray(im)

    image_list[i]=x
    i=i+1

print image_list
np.save("Test.npy",image_list)
