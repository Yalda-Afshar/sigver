#!/usr/bin/env python3

import pickle
import numpy as np
import sys
import sigver.datasets.util as util
from PIL import Image


if __name__ == "__main__":
    x, y, yforg, usermapping, filenames = util.load_dataset(sys.argv[1])
    i = int(sys.argv[2])
    imgData = x[i][0]
    img = Image.fromarray(imgData, 'L')
    img.show()

    print(imgData.shape)
    print(y[i])
    print(yforg[i])

#    for i in range(150):
#        for j in range(220):
#            bit = imgData[i][j] 
#            if bit not in [0, 1]:
#                print(bit)
    
    