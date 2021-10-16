# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 21:40:56 2021

@author: 黃乾哲
"""
from data_preprocess import content_fraction
from os import listdir

class dataloader:
    def __init__(self, path):
        self.text_list  = []
        self.label_list = []
        label_table = {"N":0,"U":0,"Y":1, "I":-1}
        fileName_list = listdir(path)
        
        for index,f in enumerate(fileName_list):
            text = self.read_file( f, path)
            self.label_list.append(label_table[f[0]])
            self.text_list.append(content_fraction(text))
        
    def read_file(self, fileName, path):
        f = open(path+fileName, mode="r",encoding="utf-8")
        text = f.read()
        f.close()
        return text
    
if __name__ == "__main__":
    path          = "D:\黃乾哲\研究所資料\課程\數位醫學\Case_Presentation_1_Data\Train_Textual\\"
    test_data = dataloader(path)
    print(test_data.label_list)
    

    
    