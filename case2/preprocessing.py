import pydicom
import cv2
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

#把dcm檔放到'data/train' 和 'data/valid'路徑下 並個別新增images資料夾
#將 pixel range縮小到0~256
df=pd.read_csv('./data/data_info.csv')
for i in os.listdir('./data/train/'):
    in_path = './data/train/'+str(i)
    # if str(i)=='images' or str(i)=='image' or str(i)=='new_image':break
    ds = pydicom.read_file(in_path) 
    img = ds.pixel_array 
    if img.max()>20000:
        img=img*(-1)+img.max()
    if img.max()<20000 and img.min()>1000:
        print(img.max(),img.min(),i)
        img=img*(-1)+img.max()
    if (img-img.min()).max()>256:
        temp=math.ceil((img-img.min()).max()/256)
        img=(img-img.min())/temp
        image = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
        # equ = cv2.equalizeHist(image)
        cv2.imwrite('./data/train/images/'+str(i[:-4])+'.jpg',image)
    else:
        img=img-img.min()
        image = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
        # equ = cv2.equalizeHist(image)
        cv2.imwrite('./data/train/images/'+str(i[:-4])+'.jpg',image)  


# for i in os.listdir('./data/valid/'):
#     in_path = './data/valid/'+str(i)
#     # if str(i)=='images' or str(i)=='image' or str(i)=='new_image':break
#     ds = pydicom.read_file(in_path) 
#     img = ds.pixel_array 
#     if (img.max()>20000) or (img.max()<20000 and img.min()>1000):
#         img=img*(-1)+img.max()
#     if (img-img.min()).max()>256:
#         temp=math.ceil((img-img.min()).max()/256)
#         img=(img-img.min())/temp
#         image = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
#         # equ = cv2.equalizeHist(image)
#         cv2.imwrite('./data/valid/images/'+str(i[:-4])+'.jpg',image)
#     else:
#         img=img-img.min()
#         image = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
#         # equ = cv2.equalizeHist(image)
#         cv2.imwrite('./data/valid/images/'+str(i[:-4])+'.jpg',image)
