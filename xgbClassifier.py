'''
A reusable class that saves a classifier with associated metadata
'''
import os
import json
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb


class xgbClassier():
    def __init__(self, model_dir: str):
        self.model = self._initialize_xgb_model()

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train our classifier with features to predict labels

        Args:
            features (pd.DataFrame): a dataframe of features--
                the components used for calculations which may be
                created from the data
            labels (pd.Series): a set of associated labels per row--
                the goal of the classifier
        """
        self.model.fit(features, labels)
        

    def predict(self, 
                features: pd.DataFrame, 
                proba: bool = False) -> np.ndarray:
        """Use a trained model to predict the output

        Args:
            features (pd.DataFrame): the input features
            proba (bool, optional): whether to return probabilities. 
                Defaults to False.

        Returns:
            np.ndarray: true or false labels for every row in features
                or probabilities of true for every row in features
        """
        
        if proba:
            return self.model.predict_proba(features)[:, 0]
        return self.model.predict(features)
    
    def assess(self, features: pd.DataFrame, labels: pd.Series) -> float:
        """Compute the accuracy of our model

        Args:
            features (pd.DataFrame): input features 
            labels (pd.Series): known labels

        Returns:
            float: the accuracy of our model
        """
        pred_labels = self.predict(features)
        return (pred_labels == labels).sum()/len(labels)
    
    def _initialize_xgb_model():
        """Create a new xgbclassifier"""
        return xgb.XGBClassifier()
