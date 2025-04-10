import os
import pandas as pd
import numpy as np
import joblib

def load_model(model_path):
    """
    Load a trained model from disk
    
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

def predict_workout_efficiency(model, input_data):
    """
    Predict workout efficiency class using the trained model
    
    Parameters:
    -----------
    model : trained model pipeline
        The trained model pipeline
    input_data : pandas.DataFrame
        Input data with features
        
    Returns:
    --------
    numpy.ndarray
        Predicted workout efficiency classes
    """
    # Make prediction
    predictions = model.predict(input_data)
    
    # Get probability estimates if available
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(input_data)
            return predictions, probabilities
        except:
            return predictions, None
    
    return predictions, None

def predict_calories_burned(model, input_data):
    """
    Predict calories burned using the trained model
    
    Parameters:
    -----------
    model : trained model pipeline
        The trained model pipeline
    input_data : pandas.DataFrame
        Input data with features
        
    Returns:
    --------
    numpy.ndarray
        Predicted calories burned
    """
    # Make prediction
    predictions = model.predict(input_data)
    
    return predictions

def create_sample_input(column_names):
    """
    Create a sample input for prediction
    
    Parameters:
    -----------
    column_names : list
        List of required column names
        
    Returns:
    --------
    pandas.DataFrame
        Sample input dataframe
    """
    # Create a dictionary with default values
    sample_data = {
        # Demographic data
        'Age': 35,
        'Gender': 'Male',
        'Height': 1.75,  # meters
        'Weight': 70,  # kg
        
        # Lifestyle metrics
        'Sleep_Hours': 7.5,
        'Water_Intake': 2.5,  # liters
        'Daily_Calories': 2200,
        'Mood': 'Happy',
        
        # Health metrics
        'Resting_Heart_Rate': 65,
        'Body_Fat_Pct': 18.5,
        'VO2_Max': 45.0,
        
        # Activity metrics
        'Workout_Type': 'Running',
        'Workout_Duration': 45,  # minutes
        'Heart_Rate': 145,  # bpm during workout
        'Steps': 8000,
        'Distance': 5.5  # km
    }
    
    # Filter to only include required columns
    filtered_data = {col: sample_data.get(col, 0) for col in column_names if col in sample_data}
    
    # For missing columns, fill with zeros (numeric) or 'Unknown' (categorical)
    for col in column_names:
        if col not in filtered_data:
            # Try to infer type from column name
            if any(hint in col.lower() for hint in ['age', 'rate', 'pct', 'hours', 'duration', 'calories', 'steps', 'distance', 'intake', 'height', 'weight']):
                filtered_data[col] = 0.0
            else:
                filtered_data[col] = 'Unknown'
    
    # Create dataframe
    return pd.DataFrame([filtered_data])

def main():
    """
    Main function to demonstrate model inference
    """
    # Load best models
    classification_model_path = 'results/models/efficiency_classifier_rf.joblib'
    regression_model_path = 'results/models/calories_burned_regressor_rf.joblib'
    
    clf_model = load_model(classification_model_path)
    reg_model = load_model(regression_model_path)
    
    if clf_model is None or reg_model is None:
        print("Failed to load models. Please run the analysis pipeline first.")
        return
    
    # Get feature names from models
    try:
        clf_feature_names = clf_model.feature_names_in_
    except:
        # If feature names not available, use a default set
        clf_feature_names = [
            'Age', 'Gender', 'Height', 'Weight', 'Sleep_Hours', 'Water_Intake',
            'Daily_Calories', 'Mood', 'Resting_Heart_Rate', 'Body_Fat_Pct',
            'VO2_Max', 'Workout_Type', 'Workout_Duration', 'Heart_Rate',
            'Steps', 'Distance'
        ]
    
    # Create sample input
    sample_input = create_sample_input(clf_feature_names)
    print("\nSample input data:")
    print(sample_input)
    
    # Make predictions
    print("\nPredicting workout efficiency...")
    efficiency_pred, efficiency_prob = predict_workout_efficiency(clf_model, sample_input)
    
    print(f"Predicted workout efficiency: {efficiency_pred[0]}")
    if efficiency_prob is not None:
        print(f"Prediction probabilities: {efficiency_prob[0]}")
    
    print("\nPredicting calories burned...")
    calories_pred = predict_calories_burned(reg_model, sample_input)
    print(f"Predicted calories burned: {calories_pred[0]:.2f}")

if __name__ == "__main__":
    main()