import os
import re
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from pycaret.classification import *

'''Clean data'''
nltk.download('stopwords')
stop_word = nltk.corpus.stopwords.words('english')

cnt=0
word_list_U=[]
word_list_Y=[]

for i in os.listdir('./Data/Train_Textual/')[:200]:
    temp=set()
    with open('./Data/Train_Textual/'+str(i),'r') as f:
        for j in f:
            for s in j.lower().replace('\n','').split(' '):
                s=re.sub(r'[^\w\s]','',s)
                if (s not in stop_word) and (s.isdigit()==False):
                    temp.add(s)
    for i in temp:
        word_list_U.append(i)

for i in os.listdir('./Data/Train_Textual/')[200:]:
    temp=set()
    with open('./Data/Train_Textual/'+str(i),'r') as f:
        for j in f:
            for s in j.lower().replace('\n','').split(' '):
                s=re.sub(r'[^\w\s]','',s)
                if (s not in stop_word) and (s.isdigit()==False):
                    temp.add(s)
    for i in temp:
        word_list_Y.append(i)

#Count the number of words
c = Counter(word_list_U)
un_mention=c.most_common()
c = Counter(word_list_Y)
mention=c.most_common()

#Filter data
cnt=0
key_word=[]
for i in range(len(mention)):
    for j in range(len(un_mention)):
        if mention[i][0] == un_mention[j][0]:
            x1=mention[i][1]/len(word_list_Y)
            x2=un_mention[j][1]/len(word_list_U)
            if (x2<x1) and (x2*4<x1) and (x1>1e-5) and (x2>1e-5):
                # print(mention[i][0],mention[i][1],un_mention[j][1])
                key_word.append(mention[i][0])
                cnt+=1


#Define columns name 
columns_name=[]
for i in key_word:
    columns_name.append(i)


'''Ceate new data'''
#Train set
new_data=np.zeros(shape=(400,len(key_word)),dtype=int)
train_df=pd.DataFrame(data=new_data,columns=columns_name)
label=[]
cnt=0
for i in os.listdir('./Data/Train_Textual/'):
    with open('./Data/Train_Textual/'+str(i),'r') as f:
        for j in f:
            h=j.lower().replace('\n','').split(' ')
            for s in h:
                s=re.sub(r'[^\w\s]','',s)
                if s in key_word:
                    train_df.iloc[cnt,key_word.index(s)]=1
    if cnt<200:
        label.append(0)
    else:
        label.append(1)
    f.close()
    cnt+=1
train_df['label']=label

#Test set
new_data=np.zeros(shape=(400,len(key_word)),dtype=int)
test_df=pd.DataFrame(data=new_data,columns=columns_name)
label=[]
cnt=0
for i in os.listdir('./Data/Test_Intuitive/'):
    with open('./Data/Test_Intuitive/'+str(i),'r') as f:
        for j in f:
            h=j.lower().replace('\n','').split(' ')
            for s in h:
                s=re.sub(r'[^\w\s]','',s)
                if s in key_word:
                    test_df.iloc[cnt,key_word.index(s)]=1
    if cnt<200:
        label.append(0)
    else:
        label.append(1)
    f.close()
    cnt+=1
test_df['label']=label


#Validation set
new_data=np.zeros(shape=(50,len(key_word)),dtype=int)
valid_df=pd.DataFrame(data=new_data,columns=columns_name)
label=[]
cnt=0
for i in os.listdir('./Data/Validation/'):
    with open('./Data/Validation/'+str(i),'r') as f:
        for j in f:
            h=j.lower().replace('\n','').split(' ')
            for s in h:
                s=re.sub(r'[^\w\s]','',s)
                if s in key_word:
                    valid_df.iloc[cnt,key_word.index(s)]=1
    cnt+=1

print("Train set shape : ",train_df.shape,", Test set shape : ",test_df.shape,", Validation set shape : ",valid_df.shape)


#Training
print("Start training : ")
exp_name = setup(data = train_df , test_data=test_df , session_id=88 , silent=True ,  target = 'label')
best = compare_models()


#Predict
model=create_model('rf')
pred_holdout = predict_model(model,data=valid_df)

pred_df=pd.read_csv('./sample_submission.csv')
pred_df['Obesity']=pred_holdout['Label']
print(pred_df.head())
# df.to_csv('./sub1007_1.csv',index=False)