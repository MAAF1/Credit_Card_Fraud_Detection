import pandas as pd
import numpy as np
from imblearn.over_sampling import *
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(data_path):
    df = pd.read_csv(data_path)
    x = df.drop(['Class'],axis=1).to_numpy()
    y = df['Class'].to_numpy()
    return x,y

def balance_the_data(X, y, balance_type = 'SMOTE',k = 3, random_state = 42, sampling_strategy = 0.05):
    
    if balance_type == 'SMOTE' :
        sampler = SMOTE(random_state=random_state, k_neighbors=k,sampling_strategy=sampling_strategy)
    elif balance_type == 'SVMSMOTE':
        sampler = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=k)
    elif balance_type == 'SMOTEENN':
        sampler = SMOTEENN(sampling_strategy=sampling_strategy,
                           smote=SMOTE(
                                        random_state=random_state,
                                        k_neighbors=k
                                        )
                           )
    elif balance_type == 'SMOTETomek': 
        sampler = SMOTETomek(sampling_strategy=sampling_strategy,
                             smote=SMOTE(random_state=random_state,
                                        k_neighbors=k)
                                        )
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def scale_data(data, scaler = 'S'): 
    if scaler == 'S':
        scaler = StandardScaler()
    elif scaler == 'R':
        scaler = RobustScaler()
    elif scaler == 'M':
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data)
    return X_scaled

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler='R'):
        self.scaler = scaler
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return scale_data(X, self.scaler)   


    

