# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:30:38 2020

@author: Gerges_Hanna
"""
#Abanoub Kamal - Abanoub Magdy - Gerges Hanna
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time

class CamiraConfig:
    #this is constructor function
    def __init__(self,feature_extractor,RF_model,labelEncoder):#Abanoub Magdy
        self.feature_extractor=feature_extractor
        self.RF_model=RF_model
        self.labelEncoder=labelEncoder
    
    #check the result and accuracy of the image 
    def __Config(self,img1):
        #Abanoub Magdy
        size1=224
        img1 = cv2.resize(img1, (size1, size1))
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        test_images1=img1
        test_images1 = np.array(test_images1)
        x_test1=test_images1
        x_test1=x_test1/255.0
        img=x_test1
        plt.imshow(img)
        input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
        input_img_features=self.feature_extractor.predict(input_img)
        # =================================== old way to get accuracy==========================================
        #Gerges
        trees = self.RF_model.estimators_
        # Get all 50 tree predictions for the first instance
        preds_for_0 = [tree.predict(input_img_features.reshape(1, -1))[0] for tree in trees]
        # print(preds_for_0)
        # =============================================================================
        prediction_RF = self.RF_model.predict(input_img_features)[0]   #Reverse the label encoder to original name
        # =============================================================================
        # ==========================Accuaracy===================================================
        #Gerges
        trees = self.RF_model.estimators_
        # Get all 50 tree predictions for the first instance
        preds_for_0 = [tree.predict(input_img_features.reshape(1, -1))[0] for tree in trees]
        acc=np.array(preds_for_0)
        ac=(np.count_nonzero(acc==prediction_RF)/acc.size)*100
        # =============================================================================
        prediction_RF = self.labelEncoder.inverse_transform([prediction_RF])
        return prediction_RF,ac
    
    #used to predcit the name name of the image and the accuracy by using image.
    def get_image_result(self,img_path):
        img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        prediction_RF,acc=self.__Config(img1)
        return prediction_RF[0],acc
    # =============================================================================
    
    # # =============================================================================
    #used to predcit the name name of the iamge and the accuracy by using video.
    def __get_Video_result(self,frame): #Abanoub Kamal
        img1 = frame
        prediction_RF,acc=self.__Config(img1)
        return prediction_RF,acc
    # =============================================================================
    
    # =============================================================================
    def Run_Camera(self,DetectCam=0): #Abanoub kamal
        cap=cv2.VideoCapture(DetectCam)
        #video
        while True:
            ret,frame=cap.read()
            result,acc=self.__get_Video_result(frame)
            print("The prediction for this image is: ", result) 
            cv2.putText(frame,"L: "+str(result[0])+" acc:"+str(acc),(10,50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2,cv2.LINE_AA)
            cv2.imshow('frame',frame)
            if cv2.waitKey(10)==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
