from credit_fraud_utils_data import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split , GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler 
from datetime import datetime
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.utils import *

def logistic_base_line(data_path, balance):
    model_name = 'Logistic Regression'
    X_train, y_train = load_data(load_config())
    X_train, y_train = balance_the_data(X_train, y_train, 'SMOTETomek')
    X_train = scale_data(X_train,scaler='R')
    scaler = 'RobustScaler'
    sampler = 'SMOTETomek'
    model = LogisticRegression(solver='saga',penalty='elasticnet',l1_ratio=0.5,max_iter=500)
    model.fit(X_train, y_train)
    params = model.get_params()
    y_pd = model.predict(X_train)
    score = accuracy_score(y_train, y_pd)
    X_val, y_val = load_data('data/val.csv')
    X_val = scale_data(X_val)
    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params":params,
        "scaler":scaler,
        "sampler":sampler,
        "best_score": score,
        "Validation_scores":{"best_precision": precision,
                             "best_f1":f1,
                              'best_recall':recall}}
    save_model_results_with_params(results,'saves/metadata/baseline.json')
    

def xgboost():
    model_name = 'XGBOOST'
    X_train, y_train = load_data('data/train.csv')
    X_train, y_train = balance_the_data(X_train, y_train, 'SMOTETomek')
    X_train = scale_data(X_train,scaler='R')
    scaler = 'RobustScaler'
    sampler = 'SMOTETomek'
    model = xgb.XGBClassifier(booster = 'gbtree', n_estimators = 200, learning_rate = 0.01, max_depth = 7)
    model.fit(X_train, y_train)
    params = model.get_params()
    y_pd = model.predict(X_train)
    score = accuracy_score(y_train, y_pd)
    X_val, y_val = load_data('data/val.csv')
    X_val = scale_data(X_val)
    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params":params,
        "scaler":scaler,
        "sampler":sampler,
        "best_score": score,
        "Validation_scores":{"best_precision": precision,
                             "best_f1":f1,
                              'best_recall':recall}}
    save_model_results_with_params(results,'saves/metadata/baseline.json')

def Randomforestbaseline():
    model_name = 'Randomforest'
    X_train, y_train = load_data('data/train.csv')
    X_train, y_train = balance_the_data(X_train, y_train, 'SMOTETomek')
    X_train = scale_data(X_train,scaler='R')
    scaler = 'RobustScaler'
    sampler = 'SvmSmote'
    model = RandomForestClassifier(n_estimators=200, max_depth=9)
    model.fit(X_train, y_train)
    params = model.get_params()
    y_pd = model.predict(X_train)
    score = accuracy_score(y_train, y_pd)
    X_val, y_val = load_data('data/val.csv')
    X_val = scale_data(X_val)
    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params":params,
        "scaler":scaler,
        "sampler":sampler,
        "best_score": score,
        "Validation_scores":{"best_precision": precision,
                             "best_f1":f1,
                              'best_recall':recall}}
    save_model_results_with_params(results,'saves/metadata/baseline.json')
def mlpclassifierbaseline():
    model_name = 'MLP'
    X_train, y_train = load_data('data/train.csv')
    X_train, y_train = balance_the_data(X_train, y_train, 'SMOTETomek')
    X_train = scale_data(X_train,scaler='R')
    scaler = 'RobustScaler'
    sampler = 'Random Under Sampler'
    model = MLPClassifier(solver='adam',hidden_layer_sizes=[30,20,50],learning_rate='constant',learning_rate_init=0.01, max_iter=1000)
    model.fit(X_train, y_train)
    params = model.get_params()
    y_pd = model.predict(X_train)
    score = accuracy_score(y_train, y_pd)
    X_val, y_val = load_data('data/val.csv')
    X_val = scale_data(X_val)
    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params":params,
        "scaler":scaler,
        "sampler":sampler,
        "best_score": score,
        "Validation_scores":{"best_precision": precision,
                             "best_f1":f1,
                              'best_recall':recall}}
    save_model_results_with_params(results,'saves/metadata/baseline.json')

if __name__ == "__main__":
    #xgboost()
    logistic_base_line()
    #Randomforestbaseline()
    #mlpclassifierbaseline()

    