import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

def create_classification_model(model_type='rf', random_state=42):
    """
    Create a classification model based on specified type
    
    Parameters:
    -----------
    model_type : str
        Type of model ('lr' for Logistic Regression, 'rf' for Random Forest)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model
        Initialized model object
    """
    if model_type == 'lr':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError(f"Invalid model type: {model_type}. Choose from 'lr' or 'rf'.")
    
    return model

def create_regression_model(model_type='rf', random_state=42):
    """
    Create a regression model based on specified type
    
    Parameters:
    -----------
    model_type : str
        Type of model ('lr' for Linear Regression, 'rf' for Random Forest)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model
        Initialized model object
    """
    if model_type == 'lr':
        model = LinearRegression()
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=random_state)
    else:
        raise ValueError(f"Invalid model type: {model_type}. Choose from 'lr' or 'rf'.")
    
    return model

def train_model(model, preprocessor, X_train, y_train):
    """
    Train a model with a preprocessing pipeline
    
    Parameters:
    -----------
    model : sklearn-compatible model
        The model to train
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Trained pipeline with preprocessing and model
    """
    # Create full pipeline with preprocessing
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    full_pipeline.fit(X_train, y_train)
    
    return full_pipeline

def evaluate_regression_model(pipeline, X_test, y_test):
    """
    Evaluate a regression model
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regression Model Evaluation:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred
    }

def evaluate_classification_model(pipeline, X_test, y_test):
    """
    Evaluate a classification model
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Classification Model Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    print("  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'report': report,
        'y_pred': y_pred
    }

def save_model(model, model_name, model_dir='results/models'):
    """
    Save a trained model to disk
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline
    model_name : str
        Name for the saved model
    model_dir : str
        Directory to save the model
        
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = f"{model_dir}/{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_path):
    """
    Load a saved model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    model
        Loaded model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
