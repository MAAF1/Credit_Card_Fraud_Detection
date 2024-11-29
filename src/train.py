from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score, classification_report, balanced_accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
import datetime
import os
import sys
from credit_fraud_utils_data import *
from utils import *
from credit_fraud_utils_eval import *
import argparse
def logistic_regression(data_config, train_config=None,cross_val=True, save_the_model=False):
    model_name = 'Logistic Regression'
    
    X_train_org, y_train_org = load_data(data_config['dataset']['train']['path'])
    X_val_org, y_val_org = load_data(data_config['dataset']['val']['path'])
    X_test,y_test = load_data(data_config['dataset']['val']['path'])

    if train_config == None and cross_val == True:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        X_train_tun, X_val_tun, y_train_tun, y_val_tun = train_test_split(X_train,y_train,test_size=0.3,shuffle=False)
        steps = [('scaler',CustomScaler()),
        ('classifier', LogisticRegression(solver='saga',penalty='elasticnet'))
        ]
        pipeline = Pipeline(steps)
        param_dist= { 
            'classifier__C':[0.1, 1.0, 10.0],
            'classifier__l1_ratio':[0,0.5,1],
            'classifier__max_iter':[1500, 4000, 7000]}
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        search =GridSearchCV(pipeline, param_dist, cv=cv, n_jobs=-1, scoring='balanced_accuracy', verbose=2)
        search.fit(X_train_tun, y_train_tun)
        best_params = search.best_params_
        score = search.best_score_
        print(f'best parameters are {best_params} with score = {score}')
        model = search.best_estimator_
        val_preds = model.predict(X_val_tun)
        f1 = f1_score(y_val_tun, val_preds)
        precision = precision_score(y_val_tun, val_preds)
        recall = recall_score(y_val_tun, val_preds)
        print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
        
        config = {"model_name":model_name,
                  "params":best_params}
        
        final_model = make_pipeline(CustomScaler(), LogisticRegression(solver='saga',penalty='elasticnet',C=best_params['classifier__C'],l1_ratio=best_params['classifier__l1_ratio'],max_iter=best_params['classifier__max_iter']))
        final_model.fit(X_train,y_train)
        y_pd = final_model.predict(X_test)
        y_pd_prob = final_model.predict_proba(X_test)
        f1_train = f1_score(y_train,final_model.predict(X_train))
        f1_test = f1_score(y_test, y_pd)
        print(f"train_f1: {f1_train}\nf1_test: {f1_test}")
        clf_report = classification_report(y_test, y_pd)
        print(clf_report)
        save_config(config)
        if save_the_model == True:
            save_model(final_model,model_name)
    else:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        params = train_config[model_name]
        model = make_pipeline(CustomScaler(), LogisticRegression(**params))
        model.fit(X_train, y_train)
        if save_the_model == True:
            save_model(model, model_name)
        return model, model_name
    
        
        

def xgboost(data_config, train_config=None,cross_val=True,save_the_model = False):
    model_name = 'XGBOOST'
    X_train_org, y_train_org = load_data(data_config['dataset']['train']['path'])
    X_val_org, y_val_org = load_data(data_config['dataset']['val']['path'])
    X_test,y_test = load_data(data_config['dataset']['val']['path'])
    if train_config == None and cross_val == True:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        X_train_tun, X_val_tun, y_train_tun, y_val_tun = train_test_split(X_train,y_train,test_size=0.3,shuffle=False)
        steps = [('scaler',CustomScaler()),
            ('classifier', xgb.XGBClassifier(eval_metric='logloss'))
            ]
        pipeline = Pipeline(steps)
        param_dist= { 
            'classifier__max_depth': [3,5,7],
            'classifier__n_estimators':[100, 200, 300],
            'classifier__colsample_bytree': [0.8, 1.0],
            'classifier__gamma':[0.8, 1.0],
            'classifier__min_child_weight': np.arange(1, 6),
            'classifier__learning_rate':[0.01, 0.1, 0.2]}
        
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        search =RandomizedSearchCV(pipeline, param_dist, cv=cv, n_jobs=-1, n_iter=40, scoring='balanced_accuracy', verbose=2)
        search.fit(X_train_tun, y_train_tun)
        best_params = search.best_params_
        score = search.best_score_
        print(f'best parameters are {best_params} with score = {score}')
        model = search.best_estimator_
        val_preds = model.predict(X_val_tun)
        f1 = f1_score(y_val_tun, val_preds)
        precision = precision_score(y_val_tun, val_preds)
        recall = recall_score(y_val_tun, val_preds)
        print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
        results = {
                    "model_name": model_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "best_params":best_params,
                    "best_score": score,
                    "Validation_scores":{"best_precision": precision,
                    "best_f1":f1,
                    'best_recall':recall}}
        config = {"model_name":model_name,
                  "params":best_params}
        save_config(config)
        final_model = make_pipeline(CustomScaler(),xgb.XGBClassifier(n_estimators=best_params['classifier__n_estimators'], min_child_weight=best_params['classifier__min_child_weight'], max_depth=best_params['classifier__max_depth'], learning_rate=best_params['classifier__learning_rate'], gamma=best_params['classifier__gamma'], colsample_bytree=best_params['classifier__colsample_bytree'], eval_metric='logloss'))
        final_model.fit(X_train,y_train)
        y_pd = final_model.predict(X_test)
        y_pd_prob = final_model.predict_proba(X_test)
        f1_train = f1_score(y_train,final_model.predict(X_train))
        f1_test = f1_score(y_test, y_pd)
        print(f"train_f1: {f1_train}\nf1_test: {f1_test}")
        clf_report = classification_report(y_test, y_pd)
        print(clf_report)
        if save_the_model == True:
            save_model(final_model,model_name)
    else:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        params = train_config[model_name]
        model = make_pipeline(CustomScaler(), xgb.XGBClassifier(**params))
        model.fit(X_train, y_train)
        if save_the_model == True:
            save_model(model, model_name)
        return model, model_name


def random_forest(data_config, train_config=None,cross_val=True,save_the_model=False):
    model_name = 'Random Forest'
    X_train_org, y_train_org = load_data(data_config['dataset']['train']['path'])
    X_val_org, y_val_org = load_data(data_config['dataset']['val']['path'])
    X_test,y_test = load_data(data_config['dataset']['val']['path'])
    if train_config == None and cross_val == True:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        X_train_tun, X_val_tun, y_train_tun, y_val_tun = train_test_split(X_train,y_train,test_size=0.3,shuffle=False)
        param_dist= { 
            'max_depth': [7,9,11],
            'n_estimators':[100, 200, 300],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]}
        
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        search =RandomizedSearchCV(RandomForestClassifier(), param_dist, cv=cv, n_jobs=-1, n_iter=40, scoring='balanced_accuracy', verbose=2)
        search.fit(X_train_tun, y_train_tun)
        best_params = search.best_params_
        score = search.best_score_
        print(f'best parameters are {best_params} with score = {score}')
        model = search.best_estimator_
        val_preds = model.predict(X_val_tun)
        f1 = f1_score(y_val_tun, val_preds)
        precision = precision_score(y_val_tun, val_preds)
        recall = recall_score(y_val_tun, val_preds)
        print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
        config = {"model_name":model_name,
                  "params":best_params}
        save_config(config)
        final_model = RandomForestClassifier(**best_params)
        final_model.fit(X_train,y_train)
        y_pd = final_model.predict(X_test)
        y_pd_prob = final_model.predict_proba(X_test)
        f1_train = f1_score(y_train,final_model.predict(X_train))
        f1_test = f1_score(y_test, y_pd)
        print(f"train_f1: {f1_train}\nf1_test: {f1_test}")
        clf_report = classification_report(y_test, y_pd)
        print(clf_report)
        if save_the_model == True:
            save_model(final_model,model_name)
    else:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTEENN')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        params = train_config[model_name]
        model = RandomForestClassifier(**params,verbose=2)
        model.fit(X_train, y_train)
        if save_the_model == True:
            save_model(model, model_name)
        return model, model_name

def neural_network(data_config, train_config=None,cross_val=True,save_the_model=False):
    model_name = 'Neural Network'
    X_train_org, y_train_org = load_data(data_config['dataset']['train']['path'])
    X_val_org, y_val_org = load_data(data_config['dataset']['val']['path'])
    X_test,y_test = load_data(data_config['dataset']['val']['path'])
    if train_config == None and cross_val == True:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        X_train_tun, X_val_tun, y_train_tun, y_val_tun = train_test_split(X_train,y_train,test_size=0.3,shuffle=False)
        param_dist= { 
            'classifier__hidden_layer_sizes': [(100, 50,60),(40,60,70,100),(50,100),(100,50)],
            'classifier__activation': ['relu', 'tanh', 'logistic'],
            'classifier__solver': ['adam', 'sgd'],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'classifier__max_iter': [700, 1000, 1500]}
        steps = [('scaler',CustomScaler()),
            ('classifier', MLPClassifier())
            ]
        pipeline = Pipeline(steps)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        search =RandomizedSearchCV(pipeline, param_dist, cv=cv, n_jobs=-1, n_iter=50, scoring='balanced_accuracy', verbose=2)
        search.fit(X_train_tun, y_train_tun)
        best_params = search.best_params_
        score = search.best_score_
        print(f'best parameters are {best_params} with score = {score}')
        model = search.best_estimator_
        val_preds = model.predict(X_val_tun)
        f1 = f1_score(y_val_tun, val_preds)
        precision = precision_score(y_val_tun, val_preds)
        recall = recall_score(y_val_tun, val_preds)
        print(f"best model scores : f1: {f1} \n precision {precision}\n recall: {recall}")
        config = {"model_name":model_name,
                  "params":best_params}
        
        final_model = make_pipeline(CustomScaler(), MLPClassifier(hidden_layer_sizes=best_params['classifier__hidden_layer_sizes'], activation=best_params['classifier__activation'],solver=best_params['classifier__solver'],alpha=best_params['classifier__alpha'],learning_rate=best_params['classifier__learning_rate'], max_iter=best_params['classifier__max_iter']))
        final_model.fit(X_train,y_train)
        y_pd = final_model.predict(X_test)
        y_pd_prob = final_model.predict_proba(X_test)
        f1_train = f1_score(y_train,final_model.predict(X_train))
        f1_test = f1_score(y_test, y_pd)
        print(f"train_f1: {f1_train}\nf1_test: {f1_test}")
        clf_report = classification_report(y_test, y_pd)
        print(clf_report)
        save_config(config)
        if save_the_model == True:
            save_model(final_model,model_name)
    else:
        X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
        X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
        params = train_config[model_name]
        model = make_pipeline(CustomScaler(), MLPClassifier(**params))
        model.fit(X_train, y_train)
        if save_the_model == True:
            save_model(model, model_name)
        return model, model_name
    

def voting_classifier(data_config, train_config, save_the_model=False):
    model_name = 'Voting Classifier'
    X_train_org, y_train_org = load_data(data_config['dataset']['train']['path'])
    X_val_org, y_val_org = load_data(data_config['dataset']['val']['path'])
    X_test,y_test = load_data(data_config['dataset']['val']['path'])
    X_train, y_train = balance_the_data(X_train_org, y_train_org, 'SMOTE')
    X_train, y_train = np.r_[X_train_org,X_val_org], np.r_[y_train_org, y_val_org]
    random_forest_params = train_config['Random Forest']
    nn_params = train_config['Neural Network']
    xgboost_params = train_config['XGBOOST']
    nn_model = make_pipeline(RobustScaler(),MLPClassifier(**nn_params))
    xgb_model = make_pipeline(RobustScaler(),xgb.XGBClassifier(**xgboost_params))
    rf_model = RandomForestClassifier(**random_forest_params)
    model = EnsembleVoteClassifier(clfs=[nn_model,
                                         xgb_model,
                                         rf_model],
                                        voting='soft',verbose=1,weights=[1, 3, 1])
    model.fit(X_train,y_train)
 
    
    if save_the_model == True:
        save_model(model, model_name)
    return model,  model_name

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Train different models for credit card fraud detection.')
    parser.add_argument('--model', type=str, choices=['logistic_regression', 'xgboost', 'random_forest', 'neural_network', 'voting_classifier'], required=True, help='Specify the model to train.')
    parser.add_argument('--cross-val', action='store_true', help='Enable cross-validation.')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model.')
    args = parser.parse_args()

    # Load configurations
    data_config = load_config('config/data_config.yml')
    train_config = load_config('config/models_config.yml')

    # Select and train the specified model
    if args.model == 'logistic_regression':
        model, model_name = logistic_regression(data_config, train_config=train_config, cross_val=args.cross_val, save_the_model=args.save_model)
    elif args.model == 'xgboost':
        model, model_name = xgboost(data_config, train_config=train_config, cross_val=args.cross_val, save_the_model=args.save_model)
    elif args.model == 'random_forest':
        model, model_name = random_forest(data_config, train_config=train_config, cross_val=args.cross_val, save_the_model=args.save_model)
    elif args.model == 'neural_network':
        model, model_name = neural_network(data_config, train_config=train_config, cross_val=args.cross_val, save_the_model=args.save_model)
    elif args.model == 'voting_classifier':
        model, model_name = voting_classifier(data_config, train_config, save_the_model=args.save_model)
    else:
        print("Invalid model name specified.")
        return

    # Load test data
    X_test, y_test = load_data(data_config['dataset']['test']['path'])



if __name__ == "__main__":
    main()

