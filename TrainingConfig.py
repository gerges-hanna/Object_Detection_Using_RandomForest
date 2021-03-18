# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:24:20 2020

@author: Gerges_Hanna
"""
#Gerges Hanna

import numpy as np 
#used to draw the image in the console.
import matplotlib.pyplot as plt
#used to find the path of the iamges.
import glob
#provides functions for interacting with the operating system
import os
import cv2
from sklearn import preprocessing


#used to read all the images and config it to prepare to train the machine
class TrainingConfig:
    def readImage_and_Config(self,trainingPath):
        #set comman size for all images.
        SIZE = 224
        train_images = []
        train_labels = [] 
        #the path of the traing images.
        for directory_path in glob.glob(trainingPath+"/*"):
            label = directory_path.split("\\")[-1]
            print(label)
            for img_path in glob.glob(os.path.join(directory_path, "*")):
                print(img_path)
                #used to read the image
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)   
                #used to set size of image to 224*224
                img = cv2.resize(img, (SIZE, SIZE))
                #change the color from RGB to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #aftre read the image add it to the array
                train_images.append(img)
                train_labels.append(label)
                
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        return train_labels,train_images
   
    #this function to encode the label of the iamge.
    def EncodeLabels(self,train_labels):
        le = preprocessing.LabelEncoder()
        le.fit(train_labels)
        train_labels_encode = le.transform(train_labels)
        return le,train_labels_encode
    #this function used to read the path for the test images.
    def readTestImages(self,testPath):
        testImage=[]
        for directory_path in glob.glob(testPath+"/*"):
            print(directory_path)
            testImage.append(directory_path)
        return testImage