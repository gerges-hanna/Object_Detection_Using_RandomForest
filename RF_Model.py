# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:35:17 2020

@author: Gerges_Hanna
"""
#Abanoub Rafaat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class RF_Model:
    #this is constructor function
    def __init__(self,X_for_RF,y_train):
        self.X_for_RF=X_for_RF
        self.y_train=y_train
    
    def fit(self,nEstimators=100,randomState=0):
        model = RandomForestClassifier(n_estimators = nEstimators, random_state = randomState) 
        model.fit(self.X_for_RF, self.y_train) 
        return model
    
    #this function to try many valuse to values for the random forset.
    def RF_byGridSearchCV(self,parameters):
        # provide iterables of values to be tested each parameter
        model = GridSearchCV(RandomForestClassifier(), parameters)
        model.fit(self.X_for_RF, self.y_train) 
        return model