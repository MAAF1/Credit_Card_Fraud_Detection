import os
import json 
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import yaml
import torch 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_config(config_path):
    with open(config_path,'r') as f:
        data = yaml.safe_load(f)
    if config_path == 'config/models_config.yml':
        data['Neural Network']['hidden_layer_sizes'] = tuple(data['Neural Network']['hidden_layer_sizes'])
    return data

def save_config(data, config_path='config/models_config.yml'):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, 'r') as f:
            existing_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        existing_data = {}
    
    new_key = data['model_name']
    existing_data[new_key] = data['params']

    with open(config_path, 'w') as f:
        yaml.dump(existing_data, f)

def save_model_results_with_params(data, filename):
    def numpy_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: numpy_to_json(v) for k, v in obj.items()}
        else:
            return obj

    try:
        # Check if the file exists
        if os.path.exists(filename):
            # Read existing data
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Append new data to existing data
            existing_data[f'trial_{len(existing_data)}'] = data
            
            # Write updated data back to the file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, default=numpy_to_json, ensure_ascii=False, indent=4)
            
            print(f"New data appended to {filename}")
        else:
            # If file doesn't exist, create it with the new data
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({'trial_0': data}, f, default=numpy_to_json, ensure_ascii=False, indent=4)
            
            print(f"File {filename} created with new data")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except json.JSONDecodeError:
        print(f"Invalid JSON format in file '{filename}'. Starting fresh.")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({'trial_0': data}, f, default=numpy_to_json, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 
def save_model(model, name):
    model_name = f'saves/models/{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")

def plot_model_comparison(model_comparison, path='saves/visualizations', save = True):
    model_comparison = pd.DataFrame(model_comparison).T
    plt.figure(figsize=(20, 30))
    sns.set_theme(font_scale=1.2)
    ax = sns.heatmap(model_comparison, annot=True, cmap='coolwarm', cbar=True, annot_kws={"size": 8}, fmt='.2f')
    plt.title('Model Performance Comparison', fontsize=20, pad=20)
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.show()
    if save == True:
        plt.savefig(f"{path}/models_comparison.png")


def load_model(model_path):
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    return model


    
