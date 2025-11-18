import joblib
import numpy as np
import os
from data import get_target_names

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    # Try different possible paths for the model
    possible_paths = [
        "model/wine_model.pkl",
        "../model/wine_model.pkl",
        "./wine_model.pkl"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("Wine model not found. Please run train.py first to create the model.")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    return y_pred

def predict_with_prob(X):
    """
    Predict class labels and return prediction probabilities.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        dict: Dictionary containing prediction, class name, and probabilities.
    """
    # Try different possible paths for the model
    possible_paths = [
        "model/wine_model.pkl",
        "../model/wine_model.pkl", 
        "./wine_model.pkl"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("Wine model not found. Please run train.py first to create the model.")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    target_names = get_target_names()
    
    result = {
        'prediction': int(y_pred[0]),
        'class_name': target_names[y_pred[0]],
        'probabilities': {
            target_names[i]: float(prob) 
            for i, prob in enumerate(y_proba[0])
        }
    }
    
    return result