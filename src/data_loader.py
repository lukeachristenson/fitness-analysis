import pandas as pd
import os
import numpy as np
import sys

# Add parent directory to path to import from data.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_dataset_path

def load_fitness_data(data_path=None):
    """
    Load the fitness tracker dataset
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the dataset CSV file. If None, will use default path from data module.
        
    Returns:
    --------
    pandas.DataFrame
        The loaded dataset
    """
    # If no path provided, get it from the data module
    if data_path is None:
        data_path = get_dataset_path()
        
    if data_path is None:
        print("Failed to get dataset path. Please check error messages.")
        return None
    
    try:
        data = pd.read_csv(data_path)
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def split_data(data, target_col, test_size=0.2, validation_size=0.1, random_state=42):
    """
    Split dataset into training, validation and test sets
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to split
    target_col : str
        The name of the target column
    test_size : float
        Proportion of data to use for testing
    validation_size : float
        Proportion of training data to use for validation
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split off the test set
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Then split the remaining data into train and validation
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test