from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, recall_score, precision_score, f1_score, auc, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_best_threshold(y_gt, y_prob):
    y_prob = y_prob[:, 1]
    thresholds = np.arange(0.0, 1.0, 0.01)

    # Initialize variables to store the best threshold and metric value
    best_threshold = 0.0
    best_f1 = 0.0
    # Evaluate each threshold
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        # Calculate the F1 score (or any other metric)
        f1 = f1_score(y_gt, y_pred)
    
        # Update the best threshold if the current one is better
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print(f"Optimal Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1}")
    return best_threshold

def eval_auc_precision_recall_curve(y_pred_prob, y_true):
    precision, recall, _ = precision_recall_curve(y_score=y_pred_prob,y_true=y_true)
     
    return float(auc(x=recall, y=precision))

def eval_model(model, model_name, X, y_true, get_optimal_threshold=False,plot=True, save_fig=False, path ='saves/visualizations'):
    y_prob = model.predict_proba(X)
    y_prob = y_prob[:, 1]
    if get_optimal_threshold == True:
        y_pd = eval_with_threshold(y_gt=y_true, y_prob=y_prob)
        model_name = f"{model_name} optimal threshold"
    else:
        y_pd = model.predict(X)

    clf_report = classification_report(y_true=y_true, y_pred=y_pd, output_dict=True)
    pr_auc = eval_auc_precision_recall_curve(y_pred_prob=y_prob, y_true=y_true)
    acc = accuracy_score(y_true=y_true, y_pred=y_pd)
    filtered_report = {
    'recall': clf_report['1']['recall'],
    'precision': clf_report['1']['precision'],
    'F1-score': clf_report['1']['f1-score'],
    'macro avg_f1': clf_report['macro avg']['f1-score'],
    'PR-AUC':eval_auc_precision_recall_curve(y_pred_prob=y_prob, y_true=y_true),
    'Accuracy': acc}
    if plot == True:
        clf_report_df = pd.DataFrame(filtered_report, index = [model_name])
        plt.figure(figsize=(10, 6))
        sns.heatmap(clf_report_df, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f'Classification Report of {model_name}')
        plt.ylabel('Metrics')
        plt.xlabel('Classes')
        if save_fig == True:
            plt.savefig(f"{path}/{model_name}.png")
        plt.show()
    
    




def eval_with_threshold(y_gt,y_prob, with_best_threshold=True, threshold=0.5 ):
    if with_best_threshold == True:
        thresholds = np.arange(0.0, 1.0, 0.01)

        # Initialize variables to store the best threshold and metric value
        best_threshold = 0.0
        best_f1 = 0.0
        # Evaluate each threshold
        for thresholdi in thresholds:
            y_pred = (y_prob >= thresholdi).astype(int)
            # Calculate the F1 score (or any other metric)
            f1 = f1_score(y_gt, y_pred)
    
            # Update the best threshold if the current one is better
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresholdi

        y_pd = (y_prob >= best_threshold).astype(int)
    else: 
        y_pd = (y_prob >= threshold).astype(int)
    return y_pd


def compare_models(model_names, y_pred_list, y_prob_list, y_gt):
    filtered_reports = []
    for model_name, y_pred, y_prob in zip(model_names, y_pred_list, y_prob_list):
        clf_report = classification_report(y_true=y_gt, y_pred=y_pred, output_dict=True)
    
        filtered_report = {
            'Recall': clf_report['1']['recall'],
            'Precision': clf_report['1']['precision'],
            'F1-score': clf_report['1']['f1-score'],
            'macro avg_f1': clf_report['macro avg']['f1-score'],
            'PR-AUC': eval_auc_precision_recall_curve(y_pred_prob=y_prob[:, 1], y_true=y_gt),
            }
        filtered_reports.append(filtered_report)


    clf_report_df = pd.DataFrame(filtered_reports, index=model_names)
    return clf_report_df

def get_mdoels_ready(models, model_names, X, y_gt, with_optimal_threshold=True):
    thresh = 'optimal threshold'
    new_model_names = []
    y_pds = []
    y_probs = []
    for model_name, model in zip(model_names, models):
        new_model_names.append(model_name)
        y_pd = model.predict(X)
        y_prob = model.predict_proba(X)
        y_pds.append(y_pd)
        y_probs.append(y_prob)
        if with_optimal_threshold == True:
            new_model_names.append(f"{thresh}")
            y_pd_best_threshold = eval_with_threshold(y_gt=y_gt,y_prob=y_prob[:,1])
            y_pds.append(y_pd_best_threshold)
            y_probs.append(y_prob)
    return new_model_names, y_pds, y_probs
