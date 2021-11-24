import os
import cv2
import numpy as np
import pandas as pd
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(224), # for vgg、resnet etc
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


image=torch.zeros(150,3,299,299)
df=pd.read_csv('./data/sample_submission.csv')
cnt=0
for i in df['FileID']:
    img= cv2.imread('./data/valid/he_image/'+str(i)+'.jpg')
    img=img/img.max()
    # print(img.min(),img.max())
    image[cnt]=torch.from_numpy(np.transpose(img,(2,0,1)))
    cnt+=1
    
input_tensor = preprocess(image)
model = models.resnet18(pretrained=False)

            
#Setting output layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 3)
)
model.load_state_dict(torch.load('./resnet18.pth'))
model.eval()
# print(model)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() 

batch_size=10
predict=[]
for i in range(0,150,batch_size):
    inputs = input_tensor[i:i+batch_size]
    inputs = Variable(inputs.to(device='cuda:0', dtype=torch.float))
    

    # wrap them in Variable
    outputs = model(inputs)
    # outputs[outputs>0.5]=1
    # outputs[outputs<=0.5]=0
    outputs_=outputs.data.cpu().numpy()
    pred=np.argmax(outputs_,axis=-1)
    for j in pred:
        if j==0:
            predict.append('Negative')
        elif j==1:
            predict.append('Typical')
        elif j==2:
            predict.append('Atypical')

df['Type']=predict
df.to_csv('./sub_1124.csv',index=False)