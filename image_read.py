
from tempfile import TemporaryFile
import numpy as np
from PIL import Image
import glob

outfile = TemporaryFile()
image_list = np.ones((120,150,565), dtype=np.int)
i=0
for filename in glob.glob('TrainingSpoof/*.png'): #assuming gif
    im=Image.open(filename)
    x=np.asarray(im)
    image_list[i]=x

print image_list[1].shape
np.save("TrainingSpoof.npy",image_list)
