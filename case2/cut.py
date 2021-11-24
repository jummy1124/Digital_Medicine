import pydicom
import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

for im in os.listdir('./data/train/images'):
    if im[-4:]=='.jpg':
        img = cv2.imread('./data/train/images/'+str(im))
        img=img-img.min()
        threshold=0.9*(img.max()-img.min())
        print(threshold,img.max(),img.min(),im)
        cnt=0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j,0]>threshold:
                    img[i,j,:]=0
                    cnt+=1
                else:
                    img[i,j,:]=img[i,j,:]*(256/threshold)
        cv2.imwrite('./data/train/new_image/'+str(im),img)  

for im in os.listdir('./data/valid/images'):
    if im[-4:]=='.jpg':
        img = cv2.imread('./data/valid/images/'+str(im))
        img=img-img.min()
        threshold=0.9*(img.max()-img.min())
        print(threshold,img.max(),img.min(),im)
        cnt=0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j,0]>threshold:
                    img[i,j,:]=0
                    cnt+=1
                else:
                    img[i,j,:]=img[i,j,:]*(256/threshold)
        cv2.imwrite('./data/valid/new_image/'+str(im),img)  