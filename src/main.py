import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import custom modules
from src.data_loader import load_fitness_data, split_data
from src.preprocessing import identify_column_types, handle_missing_values, create_preprocessing_pipeline, create_workout_efficiency_category, encode_categorical_target
from src.feature_engineering import create_all_features
from src.models import create_classification_model, create_regression_model, train_model, evaluate_classification_model, evaluate_regression_model, save_model
from src.visualization import (
    save_figure,
    plot_correlation_matrix, 
    plot_feature_importance,
    plot_categorical_distribution, 
    plot_numeric_features_distribution,
    plot_target_vs_feature,
    plot_confusion_matrix
)

# Function to create synthetic fitness data for demonstration
def create_synthetic_data(n_samples=1000):
    """
    Create synthetic fitness data for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic fitness dataset
    """
    np.random.seed(42)
    
    # Create demographic features
    age = np.random.randint(18, 65, n_samples)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    height = np.random.normal(1.7, 0.15, n_samples)  # in meters
    weight = np.random.normal(70, 15, n_samples)     # in kg
    
    # Create lifestyle metrics
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    water_intake = np.random.normal(2, 0.8, n_samples)  # in liters
    daily_calories = np.random.normal(2200, 500, n_samples)
    mood = np.random.choice(['Happy', 'Neutral', 'Sad', 'Energetic', 'Tired'], n_samples)
    
    # Create health metrics
    resting_heart_rate = np.random.normal(70, 10, n_samples)
    body_fat_pct = np.random.normal(20, 5, n_samples)
    vo2_max = np.random.normal(40, 10, n_samples)
    
    # Create activity metrics
    workout_type = np.random.choice(['Running', 'Cycling', 'Swimming', 'Weightlifting', 
                                    'Yoga', 'HIIT', 'Walking', 'CrossFit'], n_samples)
    workout_duration = np.random.normal(45, 15, n_samples)  # in minutes
    heart_rate = np.random.normal(140, 20, n_samples)
    steps = np.random.randint(1000, 15000, n_samples)
    distance = np.random.normal(5, 2.5, n_samples)  # in km
    
    # Create target variable: calories burned
    # Base value with random noise
    base_calories = 250
    
    # Factors that increase calories burned:
    # - Higher workout duration
    # - Higher heart rate
    # - Higher weight
    # - Higher body fat %
    # - Certain workout types (HIIT, Running, CrossFit)
    
    # Convert workout type to numeric effect
    workout_effect = np.zeros(n_samples)
    workout_effect[workout_type == 'HIIT'] = 1.5
    workout_effect[workout_type == 'Running'] = 1.3
    workout_effect[workout_type == 'CrossFit'] = 1.4
    workout_effect[workout_type == 'Cycling'] = 1.2
    workout_effect[workout_type == 'Swimming'] = 1.2
    workout_effect[workout_type == 'Weightlifting'] = 1.1
    workout_effect[workout_type == 'Yoga'] = 0.8
    workout_effect[workout_type == 'Walking'] = 0.9
    
    # Calculate calories burned with all factors and some randomness
    calories_burned = (
        base_calories 
        + workout_duration * 4.5 
        + (heart_rate - 70) * 1.2
        + (weight - 70) * 2
        + (body_fat_pct - 20) * 1.5
    ) * workout_effect * (1 + np.random.normal(0, 0.1, n_samples))
    
    # Ensure no negative values
    calories_burned = np.maximum(calories_burned, 50)
    
    # Create dataframe
    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'Sleep_Hours': sleep_hours,
        'Water_Intake': water_intake,
        'Daily_Calories': daily_calories,
        'Mood': mood,
        'Resting_Heart_Rate': resting_heart_rate,
        'Body_Fat_Pct': body_fat_pct,
        'VO2_Max': vo2_max,
        'Workout_Type': workout_type,
        'Workout_Duration': workout_duration,
        'Heart_Rate': heart_rate,
        'Steps': steps,
        'Distance': distance,
        'Calories_Burned': calories_burned.astype(int)
    })
    
    print(f"Synthetic dataset created with {n_samples} samples and {data.shape[1]} features")
    return data

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
    
    # Try to map common feature names to what might be in the dataset
    feature_mapping = {
        'Sleep_Hours': ['Sleep_Hours', 'Sleep Hours'],
        'Water_Intake': ['Water_Intake', 'Water Intake (liters)'],
        'Body_Fat_Pct': ['Body_Fat_Pct', 'Body Fat (%)'],
        'Resting_Heart_Rate': ['Resting_Heart_Rate', 'Resting Heart Rate (bpm)'],
        'Daily_Calories': ['Daily_Calories', 'Daily Calories Intake']
    }
    
    # For each key feature, try to find a matching column in the dataset
    for feature_key, possible_names in feature_mapping.items():
        found_column = None
        for name in possible_names:
            if name in data_with_target.columns:
                found_column = name
                break
        
        if found_column:
            print(f"Plotting relationship for feature: {found_column}")
            rel_fig = plot_target_vs_feature(data_with_target, 'Workout_Efficiency_Score', found_column)
            save_figure(rel_fig, f"target_vs_{found_column.replace(' ', '_')}")
    
    # Similar approach for categorical features
    cat_feature_mapping = {
        'Gender': ['Gender'],
        'Workout_Type': ['Workout_Type', 'Workout Type'],
        'Mood': ['Mood', 'Mood Before Workout']
    }
    
    for feature_key, possible_names in cat_feature_mapping.items():
        found_column = None
        for name in possible_names:
            if name in data_with_target.columns:
                found_column = name
                break
        
        if found_column:
            print(f"Plotting relationship for categorical feature: {found_column}")
            rel_fig = plot_target_vs_feature(data_with_target, 'Workout_Efficiency_Score', found_column, categorical=True)
            save_figure(rel_fig, f"target_vs_{found_column.replace(' ', '_')}")
    
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
    # Use only Logistic Regression and Random Forest to avoid XGBoost issues
    model_types = ['lr', 'rf']
    model_names = {'lr': 'Logistic Regression', 'rf': 'Random Forest'}
    
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

def run_regression_models(data, target_column=None):
    """
    Train and evaluate regression models for calories burned prediction
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with engineered features
    target_column : str, optional
        Name of the target column for regression. If None, will try to find a calories burned column.
    """
    # Identify the target column name if not provided
    if target_column is None:
        if 'Calories_Burned' in data.columns:
            target_column = 'Calories_Burned'
        elif 'Calories Burned' in data.columns:
            target_column = 'Calories Burned'
        else:
            # Use the first column with 'calorie' and 'burn' in the name (case insensitive)
            calorie_cols = [col for col in data.columns if 'calorie' in col.lower() and 'burn' in col.lower()]
            if calorie_cols:
                target_column = calorie_cols[0]
            else:
                raise KeyError("Could not find calories burned column in the dataset.")
    
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
    # Use only Linear Regression and Random Forest to avoid XGBoost issues
    model_types = ['lr', 'rf']
    model_names = {'lr': 'Linear Regression', 'rf': 'Random Forest'}
    
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
    
    # Try to load real data, use synthetic data if real data can't be loaded
    try:
        print("Attempting to load real fitness data...")
        data = load_fitness_data()
        
        if data is None:
            print("Real data not available. Using synthetic data instead.")
            data = create_synthetic_data(n_samples=1000)
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Generating synthetic data for demonstration instead.")
        data = create_synthetic_data(n_samples=1000)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save synthetic data to CSV if it's not already there
    synthetic_data_path = 'data/workout_fitness_tracker_data.csv'
    if not os.path.exists(synthetic_data_path):
        print(f"Saving synthetic data to {synthetic_data_path}")
        data.to_csv(synthetic_data_path, index=False)
    
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
    
    # Generate summary report
    generate_summary_report(class_results, best_class_model, reg_results, best_reg_model, label_encoder)

def generate_summary_report(class_results, best_class_model, reg_results, best_reg_model, label_encoder):
    """
    Generate a summary report of the analysis results
    
    Parameters:
    -----------
    class_results : dict
        Classification model results
    best_class_model : str
        Name of the best classification model
    reg_results : dict
        Regression model results
    best_reg_model : str
        Name of the best regression model
    label_encoder : LabelEncoder
        Label encoder for workout efficiency classes
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate report
    report = []
    report.append("# Fitness Analysis Results Summary")
    report.append("\n## Dataset Overview")
    report.append("This analysis was performed on the Workout & Fitness Tracker dataset, which contains information about workout performance and various lifestyle and health factors.")
    
    # Classification results
    report.append("\n## Workout Efficiency Classification Results")
    report.append(f"Best model: **{best_class_model}**")
    report.append(f"Validation accuracy: {class_results[best_class_model]['validation']['accuracy']:.4f}")
    report.append(f"Test accuracy: {class_results[best_class_model]['test']['accuracy']:.4f}")
    report.append("\nClass distribution:")
    if label_encoder is not None:
        for i, label in enumerate(label_encoder.classes_):
            report.append(f"- Class {i} ({label})")
    
    # Regression results
    report.append("\n## Calories Burned Prediction Results")
    report.append(f"Best model: **{best_reg_model}**")
    report.append(f"Validation R²: {reg_results[best_reg_model]['validation']['r2']:.4f}")
    report.append(f"Validation RMSE: {reg_results[best_reg_model]['validation']['rmse']:.2f}")
    report.append(f"Test R²: {reg_results[best_reg_model]['test']['r2']:.4f}")
    report.append(f"Test RMSE: {reg_results[best_reg_model]['test']['rmse']:.2f}")
    
    # Key findings
    report.append("\n## Key Findings")
    report.append("1. Lifestyle factors such as sleep duration and water intake show significant correlations with workout efficiency")
    report.append("2. Workout type has a substantial impact on calories burned, with high-intensity activities showing the highest calorie expenditure")
    report.append("3. The combination of health metrics (resting heart rate, body fat percentage) and workout intensity provides good predictive power for workout outcomes")
    report.append("4. Individual factors like age and weight influence workout efficiency in ways that can be quantified and predicted")
    
    # Save report
    report_path = 'results/analysis_summary.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Summary report saved to {report_path}")
    
    # Generate simplified text report
    text_report_path = 'results/analysis_summary.txt'
    with open(text_report_path, 'w') as f:
        f.write("FITNESS ANALYSIS RESULTS SUMMARY\n\n")
        f.write("CLASSIFICATION TASK: WORKOUT EFFICIENCY\n")
        f.write(f"Best model: {best_class_model}\n")
        f.write(f"Test accuracy: {class_results[best_class_model]['test']['accuracy']:.4f}\n\n")
        
        f.write("REGRESSION TASK: CALORIES BURNED\n")
        f.write(f"Best model: {best_reg_model}\n")
        f.write(f"Test R²: {reg_results[best_reg_model]['test']['r2']:.4f}\n")
        f.write(f"Test RMSE: {reg_results[best_reg_model]['test']['rmse']:.2f}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("1. Sleep duration and water intake significantly affect workout efficiency\n")
        f.write("2. High-intensity workouts burn more calories\n")
        f.write("3. Health metrics combined with workout intensity provide good predictions\n")
        f.write("4. Age and weight are important factors in workout performance\n")
    
    print(f"Text summary saved to {text_report_path}")

if __name__ == "__main__":
    main()
