"""
Data acquisition module for Fitness Analysis Project.

This module handles downloading and setting up the dataset from Kaggle.
It provides functions to download the dataset, copy it to the project data directory,
and check if the dataset is already available.
"""

import os
import shutil
import kagglehub

def download_fitness_dataset(data_dir='data'):
    """
    Download the Workout & Fitness Tracker dataset from Kaggle
    and copy it to the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
        
    Returns:
    --------
    str
        Path to the copied dataset file
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download latest version from Kaggle
    print("Downloading dataset from Kaggle...")
    try:
        kaggle_path = kagglehub.dataset_download("adilshamim8/workout-and-fitness-tracker-data")
        print(f"Dataset downloaded to: {kaggle_path}")
        
        # Find the CSV file
        csv_file = os.path.join(kaggle_path, "workout_fitness_tracker_data.csv")
        
        # Copy it to our data directory
        local_path = os.path.join(data_dir, "workout_fitness_tracker_data.csv")
        shutil.copy2(csv_file, local_path)
        print(f"Dataset copied to: {local_path}")
        
        return local_path
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def dataset_exists(data_dir='data'):
    """
    Check if the dataset file exists in the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
        
    Returns:
    --------
    bool
        True if dataset exists, False otherwise
    """
    dataset_path = os.path.join(data_dir, "workout_fitness_tracker_data.csv")
    return os.path.exists(dataset_path)

def get_dataset_path(data_dir='data'):
    """
    Get the path to the dataset file, downloading it if necessary.
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
        
    Returns:
    --------
    str
        Path to the dataset file
    """
    if dataset_exists(data_dir):
        return os.path.join(data_dir, "workout_fitness_tracker_data.csv")
    else:
        return download_fitness_dataset(data_dir)

if __name__ == "__main__":
    # If script is run directly, download the dataset
    path = get_dataset_path()
    if path:
        print(f"Dataset is available at: {path}")
    else:
        print("Failed to get dataset. Please check the error messages.")