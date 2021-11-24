import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

for i in os.listdir('./data/train/images/'):
    print(i)
    img = cv.imread('./data/train/images/'+str(i),0)
    equ = cv.equalizeHist(img)
    cv.imwrite('./data/train/he_image/'+str(i),equ)

for i in os.listdir('./data/valid/images/'):
    print(i)
    img = cv.imread('./data/valid/images/'+str(i),0)
    equ = cv.equalizeHist(img)
    cv.imwrite('./data/valid/he_image/'+str(i),equ)