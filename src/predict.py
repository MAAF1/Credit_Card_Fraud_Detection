from utils import *

def get_data_X_and_y(config_path, mode='test'):
    data = load_config(config_path=config_path)
    X, y = data['dataset'][mode]['path']
    return X, y

def make_custom_input(X):
    return X


def get_single_model(model_path, model_name):
    model = load_model(model_path=model_path)
    return model, model_name

def predict(model, X, mode='proba'):
    y_pd = model.predict(X)
    if mode == 'proba':
        y_proba = model.predict_proba(X)
        return y_pd, y_proba
    else:
        return y_pd
    
