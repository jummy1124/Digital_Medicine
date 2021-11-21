import cv2
import os
import numpy as np
import pandas as pd
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torchvision import transforms
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


preprocess = transforms.Compose([
    transforms.Resize(224), # for vgg、resnet etc
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image=torch.zeros(1200,3,299,299)
label=np.zeros(shape=(1200,))
df=pd.read_csv('./data/data_info.csv')

for i in range(df.shape[0]):
    # print('./data/train/images/'+str(df['FileID'][i])+'.jpg')
    img= cv2.imread('./data/train/images/'+str(df['FileID'][i])+'.jpg')
    img=img/img.max()
    image[i]=torch.from_numpy(np.transpose(img,(2,0,1)))
    if df['Negative'][i]==1:
        label[i]=0
    elif df['Typical'][i]==1:
        label[i]=1
    elif df['Atypical'][i]==1:
        label[i]=2
        

input_tensor = preprocess(image)
model = models.resnet18(pretrained=True)

#Freeze layers
cnt=0
for child in model.children(): 
    cnt+=1
    for param in child.parameters():
        if cnt<8:
            param.requires_grad = False


#Setting output layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 3)
)
# print(model)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() 

X_train, X_test, y_train, y_test = train_test_split(input_tensor,label , test_size=0.3, random_state=17)

batch_size=10
def evaluate(model):
    model.eval()
    acc=0
    f1=0
    f1_1=0
    testing_loss=0.0
    for i in range(0,X_test.shape[0],batch_size):
        inputs = X_test[i:i+batch_size]
        inputs = Variable(inputs.to(device='cuda:0', dtype=torch.float))
        labels = torch.FloatTensor(np.array(y_test[i:i+batch_size]))

        # wrap them in Variable
        inputs, labels = Variable(inputs.to(device='cuda:0', dtype=torch.float)), Variable(labels.to(device='cuda:0', dtype=torch.float))
        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_func(outputs, labels.long())
        testing_loss += loss.item()
        outputs_=outputs.data.cpu().numpy()
        labels_=labels.data.cpu().numpy()

        f1+=f1_score(labels_, np.argmax(outputs_,axis=-1), average='macro')
        acc+=accuracy_score(labels_, np.argmax(outputs_,axis=-1))

    return f1/(X_test.shape[0]/batch_size),testing_loss/(X_test.shape[0]/batch_size) 


epochs=50
for epoch in range(epochs):  # loop over the dataset multiple times
    print ("\nEpoch ", epoch)
    
    running_loss = 0.0
    acc=0.0
    f1=0.0
    for i in range(0,X_train.shape[0],batch_size):
        inputs = X_train[i:i+batch_size]
        inputs = Variable(inputs.to(device='cuda:0', dtype=torch.float))
        labels = torch.FloatTensor(np.array(y_train[i:i+batch_size]))
        
        # wrap them in Variable
        inputs, labels = Variable(inputs.to(device='cuda:0', dtype=torch.float)), Variable(labels.to(device='cuda:0', dtype=torch.float))
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels.long())
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()

        outputs_=outputs.data.cpu().numpy()
        labels_=labels.data.cpu().numpy()
        acc+=accuracy_score(labels_, np.argmax(outputs_,axis=-1))
        f1+=f1_score(labels_, np.argmax(outputs_,axis=-1), average='macro')
    test_f1,testing_loss=evaluate(model)

    print("Train loss:" , round(running_loss/X_train.shape[0],4) , ", Train f1 score:" , round(f1/X_train.shape[0],4) ,
          ", Test loss:" , round(testing_loss,4) , ", Test f1 score:" , round(test_f1,4))
    # result[0,epoch]=round(running_loss/X_train.shape[0],4)
    # result[1,epoch]=round(f1/X_train.shape[0],4)
    # result[2,epoch]=round(testing_loss,4)
    # result[3,epoch]=round(test_f1,4)
    if test_f1>0.55  :
        torch.save(model.state_dict(), './resnet18.pth')
        break