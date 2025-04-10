import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import custom modules
from data_loader import load_fitness_data, split_data
from preprocessing import identify_column_types, handle_missing_values, create_preprocessing_pipeline, create_workout_efficiency_category, encode_categorical_target
from feature_engineering import create_all_features
from models import create_classification_model, create_regression_model, train_model, evaluate_classification_model, evaluate_regression_model, save_model
from visualization import (
    save_figure,
    plot_correlation_matrix, 
    plot_feature_importance,
    plot_categorical_distribution, 
    plot_numeric_features_distribution,
    plot_target_vs_feature,
    plot_confusion_matrix
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_eda(data, results_dir='results'):
    """
    Run exploratory data analysis and save results
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    results_dir : str
        Directory to save results
    """
    print("\n" + "=" * 80)
    print("Starting Exploratory Data Analysis...")
    print("=" * 80)
    
    # Basic dataset information
    print(f"\nDataset Overview:")
    print(f"  Shape: {data.shape}")
    print(f"  Columns: {data.columns.tolist()}")
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(data)
    
    # Check for missing values and handle them
    print(f"\nChecking for missing values...")
    data_clean = handle_missing_values(data)
    
    # Plot numeric feature distributions
    print(f"\nPlotting numeric feature distributions...")
    num_fig = plot_numeric_features_distribution(data_clean, columns=numeric_cols[:12])  # First 12 numeric features
    save_figure(num_fig, "numeric_features_distribution")
    
    # Plot categorical feature distributions
    print(f"\nPlotting categorical feature distributions...")
    for col in categorical_cols[:5]:  # First 5 categorical features
        cat_fig = plot_categorical_distribution(data_clean, col)
        save_figure(cat_fig, f"categorical_distribution_{col}")
    
    # Plot correlation matrix
    print(f"\nPlotting correlation matrix...")
    corr_fig = plot_correlation_matrix(data_clean, figsize=(14, 12))
    save_figure(corr_fig, "correlation_matrix")
    
    # Create the workout efficiency feature
    print(f"\nCreating workout efficiency feature...")
    data_with_target = create_workout_efficiency_category(data_clean)
    
    # Plot relationships between target and key features
    print(f"\nPlotting target vs feature relationships...")
    key_numeric_features = ['Sleep_Hours', 'Water_Intake', 'Body_Fat_Pct', 'Resting_Heart_Rate', 'Daily_Calories']
    for feature in key_numeric_features:
        if feature in data_with_target.columns:
            rel_fig = plot_target_vs_feature(data_with_target, 'Workout_Efficiency_Score', feature)
            save_figure(rel_fig, f"target_vs_{feature}")
    
    key_categorical_features = ['Gender', 'Workout_Type', 'Mood']
    for feature in key_categorical_features:
        if feature in data_with_target.columns:
            rel_fig = plot_target_vs_feature(data_with_target, 'Workout_Efficiency_Score', feature, categorical=True)
            save_figure(rel_fig, f"target_vs_{feature}")
    
    print("\nExploratory Data Analysis completed successfully!")
    return data_with_target

def run_feature_engineering(data):
    """
    Run feature engineering process
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Clean dataset with basic target
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with engineered features
    """
    print("\n" + "=" * 80)
    print("Starting Feature Engineering...")
    print("=" * 80)
    
    # Apply all feature engineering transformations
    data_engineered = create_all_features(data)
    
    # Encode the categorical target for classification
    data_encoded, label_encoder = encode_categorical_target(data_engineered)
    
    print("\nFeature Engineering completed successfully!")
    return data_encoded, label_encoder

def run_classification_models(data, target_column, label_encoder=None):
    """
    Train and evaluate classification models for workout efficiency
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with engineered features
    target_column : str
        Name of the target column (must be already encoded for classification)
    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Label encoder for the target variable
    """
    print("\n" + "=" * 80)
    print("Training Classification Models for Workout Efficiency...")
    print("=" * 80)
    
    # Identify feature types for preprocessing
    feature_data = data.drop([c for c in data.columns if 'Workout_Efficiency' in c], axis=1)
    numeric_cols, categorical_cols = identify_column_types(feature_data)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data, target_column)
    
    # Get original class labels if encoder is provided
    class_labels = None
    if label_encoder is not None:
        class_labels = label_encoder.classes_.tolist()
    
    model_results = {}
    model_types = ['lr', 'rf', 'xgb']
    model_names = {'lr': 'Logistic Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
    
    # Train and evaluate all models
    for model_type in model_types:
        print(f"\nTraining {model_names[model_type]} Classifier...")
        
        # Create and train model
        model = create_classification_model(model_type, random_state=RANDOM_SEED)
        trained_pipeline = train_model(model, preprocessor, X_train, y_train)
        
        # Evaluate on validation set
        print(f"Evaluating on validation set:")
        val_results = evaluate_classification_model(trained_pipeline, X_val, y_val)
        
        # Evaluate on test set
        print(f"Evaluating on test set:")
        test_results = evaluate_classification_model(trained_pipeline, X_test, y_test)
        
        # Plot confusion matrix
        cm_fig = plot_confusion_matrix(y_test, test_results['y_pred'], labels=class_labels)
        save_figure(cm_fig, f"confusion_matrix_{model_type}")
        
        # Plot feature importance if model supports it
        try:
            # Get feature names after preprocessing (for one-hot encoded features)
            model_instance = trained_pipeline.named_steps['model']
            feature_names = numeric_cols + categorical_cols
            
            importance_fig = plot_feature_importance(model_instance, feature_names)
            save_figure(importance_fig, f"feature_importance_{model_type}")
        except Exception as e:
            print(f"Could not plot feature importance: {e}")
        
        # Save model
        model_path = save_model(trained_pipeline, f"efficiency_classifier_{model_type}")
        
        # Save results
        model_results[model_type] = {
            'validation': val_results,
            'test': test_results,
            'model_path': model_path
        }
    
    # Find best model based on validation accuracy
    best_model = max(model_results, key=lambda k: model_results[k]['validation']['accuracy'])
    
    print(f"\nBest Classification Model: {model_names[best_model]}")
    print(f"  Validation Accuracy: {model_results[best_model]['validation']['accuracy']:.4f}")
    print(f"  Test Accuracy: {model_results[best_model]['test']['accuracy']:.4f}")
    
    return model_results, best_model

def run_regression_models(data, target_column='Calories_Burned'):
    """
    Train and evaluate regression models for calories burned prediction
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with engineered features
    target_column : str
        Name of the target column for regression
    """
    print("\n" + "=" * 80)
    print(f"Training Regression Models for {target_column}...")
    print("=" * 80)
    
    # Remove efficiency-related columns that were created for classification
    cols_to_drop = [c for c in data.columns if 'Workout_Efficiency' in c]
    data_reg = data.drop(cols_to_drop, axis=1)
    
    # Identify feature types for preprocessing
    feature_data = data_reg.drop([target_column], axis=1)
    numeric_cols, categorical_cols = identify_column_types(feature_data)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data_reg, target_column)
    
    model_results = {}
    model_types = ['lr', 'rf', 'xgb']
    model_names = {'lr': 'Linear Regression', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
    
    # Train and evaluate all models
    for model_type in model_types:
        print(f"\nTraining {model_names[model_type]} Regressor...")
        
        # Create and train model
        model = create_regression_model(model_type, random_state=RANDOM_SEED)
        trained_pipeline = train_model(model, preprocessor, X_train, y_train)
        
        # Evaluate on validation set
        print(f"Evaluating on validation set:")
        val_results = evaluate_regression_model(trained_pipeline, X_val, y_val)
        
        # Evaluate on test set
        print(f"Evaluating on test set:")
        test_results = evaluate_regression_model(trained_pipeline, X_test, y_test)
        
        # Plot feature importance if model supports it
        try:
            # Get feature names after preprocessing
            model_instance = trained_pipeline.named_steps['model']
            feature_names = numeric_cols + categorical_cols
            
            importance_fig = plot_feature_importance(model_instance, feature_names)
            save_figure(importance_fig, f"feature_importance_reg_{model_type}")
        except Exception as e:
            print(f"Could not plot feature importance: {e}")
        
        # Scatter plot of actual vs predicted values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, test_results['y_pred'], alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_names[model_type]}: Actual vs Predicted')
        save_figure(fig, f"actual_vs_predicted_{model_type}")
        
        # Save model
        model_path = save_model(trained_pipeline, f"{target_column.lower()}_regressor_{model_type}")
        
        # Save results
        model_results[model_type] = {
            'validation': val_results,
            'test': test_results,
            'model_path': model_path
        }
    
    # Find best model based on validation R²
    best_model = max(model_results, key=lambda k: model_results[k]['validation']['r2'])
    
    print(f"\nBest Regression Model: {model_names[best_model]}")
    print(f"  Validation R²: {model_results[best_model]['validation']['r2']:.4f}")
    print(f"  Validation RMSE: {model_results[best_model]['validation']['rmse']:.4f}")
    print(f"  Test R²: {model_results[best_model]['test']['r2']:.4f}")
    print(f"  Test RMSE: {model_results[best_model]['test']['rmse']:.4f}")
    
    return model_results, best_model

def main():
    """
    Main function to run the fitness data analysis pipeline
    """
    print("\n" + "=" * 80)
    print("Fitness Data Analysis Pipeline")
    print("=" * 80)
    
    # Start timing
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_fitness_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Run exploratory data analysis
    data_with_target = run_eda(data)
    
    # Run feature engineering
    data_engineered, label_encoder = run_feature_engineering(data_with_target)
    
    # Run classification models for workout efficiency
    class_results, best_class_model = run_classification_models(
        data_engineered, 'Workout_Efficiency_Encoded', label_encoder
    )
    
    # Run regression models for calories burned
    reg_results, best_reg_model = run_regression_models(data_engineered)
    
    # Finish timing
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nFinished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    
    print("\n" + "=" * 80)
    print("Analysis Pipeline Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
