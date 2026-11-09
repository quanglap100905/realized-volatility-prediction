import numpy as np

def rmspe(y_true, y_pred):
    """Calculates Root Mean Square Percentage Error"""
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def feval_rmspe(preds, train_data):
    """Custom LightGBM evaluation function for RMSPE"""
    labels = train_data.get_label()
    return 'RMSPE', rmspe(labels, preds), False