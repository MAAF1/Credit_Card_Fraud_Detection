# Credit_Card_Fraud_Detection
## Overview:
### This project aims to detect fraudulent credit card transactions using machine learning models with imbalanced dataset. The project includes data preprocessing, model training, evaluation, and deployment using Streamlit.
## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#work)

## Project Structure

The project is organized as follows:

```plaintext
Credit_Card_Fraud_Detection/
├── app.py
├── config/
│   └── config.yml
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── README.md
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_training.ipynb
├── __pycache__/
├── README.md
├── requirements.txt
├── saves/
│   ├── models/
│   └── visualizations/
├── src/
│   ├── base_lines.py
│   ├── credit_fraud_utils_data.py
│   ├── credit_fraud_utils_eval.py
│   ├── __init__.py
│   ├── train.py
|   └── utils.py
```
## Installation
### To setup the project follow these steps:
#### 1- Clone the repositry 
```bash
    git clone https://github.com/yourusername/Credit_Card_Fraud_Detection.git
    cd Credit_Card_Fraud_Detection
```
 #### 2- Create a Virtual Environment (Recommended)
```bash
    python -m venv env
    source env/bin/activate  # For Windows: env\Scripts\activate
```
 #### 3- Install Required Packages
```bash
    pip install -r requirements.txt
```
## Usage
### Running the Training Script with Argument Parser
### You can specify the model to train, whether to perform cross-validation, and whether to save the model using command-line arguments.
    Train Logistic Regression Model with Cross-Validation and Save

    ```bash
    python src/train.py --model logistic_regression --cross-val --save-model
    ```
### Available models :
    logistic_regression
    xgboost
    random_forest
    neural_network
    voting_classifier

### Running the Streamlit App
    1 - Start the streamlit app
    ```bash
    streamlit run app.py
    ```
    2- Access the app
    Open a web browser and navigate to http://localhost:8501/ to interact with the application.

## Future Work

### Model Improvements

- Perform hyperparameter tuning for better model performance.

- Explore data augmentation techniques to handle the imbalanced dataset.

- Experiment with additional feature engineering techniques.

- Investigate ensemble methods to combine multiple models for improved accuracy.

- Incorporate model explainability techniques like SHAP or LIME.

