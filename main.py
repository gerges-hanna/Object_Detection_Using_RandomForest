# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:30:10 2020

@author: Gerges_Hanna
"""
import sys
sys.path.append('G:/python project/Object_Detiction_Random_forestst/')

from CamiraConfig import CamiraConfig
from TrainingConfig import TrainingConfig
from ExtractFeature import ExtractFeature
from RF_Model import RF_Model

#Gerges
trainConfig=TrainingConfig()
train_labels,train_images=trainConfig.readImage_and_Config('G:/python project/Object_Detiction_Random_forestst/BasicTraining/train')
# =============================================================================

#Gerges
# =============================================================================
le,train_labels_encode=trainConfig.EncodeLabels(train_labels)
# =============================================================================

#Abanoub Magdy
#this used to split the images and take 10% of the iamge as the test to get the total accuracy of the project
#random state means that 20 image repeated between classes.
#stratify used too make sure that the system take the correct number of image from each category.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_images,train_labels_encode,stratify=train_labels,test_size=0.1,random_state=20)

###################################################################
# Normalize pixel values to between 0 and 1 
#Gerges
x_train,x_test=x_train/255.0,x_test/255.0

# =============================================================================

#Andrew 
#this used to call the VGG16 to know the summary of what filters used in the analysis.
extract=ExtractFeature()
feature_extractor=extract.feature_extract()
print(feature_extractor.summary())
# =============================================================================



# =============================================================================
#Gerges
#Now, let us use features from convolutional network for RF
X_for_RF = feature_extractor.predict(x_train,verbose=1) #This is out X input to RF
# Save the model to the disk
extract.saveModel(X_for_RF,"G:/python project/Object_Detiction_Random_forestst","LatestModel.pkl")


# load the model from disk
X_for_RF=extract.getModel("G:/python project/Object_Detiction_Random_forestst/LatestModel.pkl")
# =============================================================================
# #RANDOM FOREST #Abanoub Raafat  

rf=RF_Model(X_for_RF,y_train)
RF_model=rf.fit(nEstimators=230,randomState=42)

# too get best prameters in random forests to use
modelGrid=rf.RF_byGridSearchCV({'n_estimators': [100,150,240],'random_state':[0,42,80]})
#print the best accuracy from all this random values
print(modelGrid.best_score_)
#print the best parameters.
print(modelGrid.best_params_)
#if we want to know the properties of this model
dir(modelGrid)

#Gerges
import pickle
# save the model to disk
filename = 'G:/python project/Object_Detiction_Random_forestst/RF_Model.sav'
pickle.dump(RF_model, open(filename, 'wb'))

 
# load the model from disk
RF_model = pickle.load(open(filename, 'rb'))
# =============================================================================
#Abanoub Magdy
X_test_feature = feature_extractor.predict(x_test)
#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(X_test_feature)

#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)

y_test2=le.inverse_transform(y_test)
#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test2, prediction_RF)) #in this model(LatestModel.pkl) equal 0.92567
# =============================================================================

#Gerges
# Too put All image in test file in array
test_images=[]
test_images=trainConfig.readTestImages("G:/python project/Object_Detiction_Random_forestst/BasicTraining/Test")



   


        


##Abanoub Kamal
camera=CamiraConfig(feature_extractor,RF_model,le)

res=camera.get_image_result(test_images[6])
print(res)

#to run the camera.
camera.Run_Camera(DetectCam=0)
# =============================================================================
