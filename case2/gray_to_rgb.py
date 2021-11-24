import os
import cv2
import numpy as np 

for im in os.listdir('./data/train/new_image'):
    print(im)
    img = cv2.imread('./data/train/new_image/'+str(im),0)
    img_R = cv2.equalizeHist(img)
    img_G = img
    img_B = cv2.bilateralFilter(img,9,75,75)
    new_img=np.zeros(shape=(299,299,3))
    new_img[:,:,0]=img_B
    new_img[:,:,1]=img_G
    new_img[:,:,2]=img_R
    cv2.imwrite('./data/train/color_image/'+str(im),new_img)

for im in os.listdir('./data/valid/new_image'):
    print(im)
    img = cv2.imread('./data/valid/new_image/'+str(im),0)
    img_R = cv2.equalizeHist(img)
    img_G = img
    img_B = cv2.bilateralFilter(img,9,75,75)
    new_img=np.zeros(shape=(299,299,3))
    new_img[:,:,0]=img_B
    new_img[:,:,1]=img_G
    new_img[:,:,2]=img_R
    cv2.imwrite('./data/valid/color_image/'+str(im),new_img)