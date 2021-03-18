# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:53:44 2020

@author: Gerges_Hanna
"""

#Andrew Emad 

from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump,load

#this function used to get the feature of each image to help in image detection.
class ExtractFeature:
    def feature_extract(self):
        feature_extractor=VGG16()
        #this mean to remove the last layer of the VGG16 which can know the name of the images, so we remove this layer to use random forest.
        feature_extractor.layers.pop()
        feature_extractor=Model(inputs=feature_extractor.inputs,outputs=feature_extractor.layers[-1].output)
        #return VGG16 after remove this layer.
        return feature_extractor
    
    #this function used to save the data of the model instead of taking long time to predict the features
    def saveModel(self,model,path,nameFile):
        dump(model, open(path+'/'+nameFile,'wb'))
    
    #this function used to return the model from the disk.
    def getModel(self,FullPath):
        model=load(open(FullPath, 'rb'))
        return model
        