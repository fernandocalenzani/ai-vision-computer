import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

# import tensorflow as tf

"""---VARS---"""
__imgPath = []
__imgs = []
__class = []
__path = "data/Datasets/homer_bart_1/homer_bart_1"
__files = [os.path.join(__path, f) for f in sorted(os.listdir(__path))]
__l, __h = 128, 128

for __imgPath in __files:
    try:
        image = cv2.imread(__imgPath)
        (H, W) = image.shape[:2]
    except:
        continue

    image = cv2.resize(image, (__l, __h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.ravel()

    __imgs.append(image)

    imgName = os.path.basename(os.path.normpath(__imgPath))

    if (imgName.startswith('b')):
        imgClass = 0
    else:
        imgClass = 1

    __class.append(imgClass)

x = np.asarray(__imgs)
y = np.asarray(__class)

