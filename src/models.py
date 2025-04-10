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

def evaluate_regression_model(pipeline, X_test, y_test, model_name='regression_model'):
    """
    Evaluate a regression model with enhanced visualizations
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    model_name : str
        Name of the model for saving visualizations
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Create directory for evaluation visualizations
    os.makedirs('results/figures/model_evaluation/regression', exist_ok=True)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics
    from sklearn.metrics import mean_absolute_error, explained_variance_score, max_error
    mae = mean_absolute_error(y_test, y_pred)
    explained_variance = explained_variance_score(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    
    print(f"Regression Model Evaluation:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Explained Variance: {explained_variance:.4f}")
    print(f"  Maximum Error: {max_err:.4f}")
    
    # Create scatter plot of actual vs predicted values with better styling
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    
    # Add a perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Add error bands (±RMSE)
    plt.fill_between([min_val, max_val], 
                    [min_val - rmse, max_val - rmse],
                    [min_val + rmse, max_val + rmse],
                    color='gray', alpha=0.2, label=f'RMSE = {rmse:.1f}')
    
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'Actual vs Predicted Values - {model_name}', fontsize=16)
    
    # Add metrics as text box
    text_box = (
        f'R² = {r2:.4f}\n'
        f'RMSE = {rmse:.2f}\n'
        f'MAE = {mae:.2f}'
    )
    plt.figtext(0.15, 0.8, text_box, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save figure
    fig = plt.gcf()
    from visualization import save_figure
    save_figure(fig, f"model_evaluation/regression/actual_vs_predicted_{model_name}")
    
    # Create residuals plot
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Predicted Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.title(f'Residual Plot - {model_name}', fontsize=16)
    
    # Add histogram of residuals as an inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(plt.gca(), width="30%", height="30%", loc=1)
    ax_inset.hist(residuals, bins=20, color='skyblue', edgecolor='black')
    ax_inset.set_title('Residuals Distribution')
    ax_inset.axvline(x=0, color='r', linestyle='--')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save residuals figure
    fig = plt.gcf()
    save_figure(fig, f"model_evaluation/regression/residuals_{model_name}")
    
    # Return comprehensive metrics
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'explained_variance': explained_variance,
        'max_error': max_err,
        'y_pred': y_pred
    }

def evaluate_classification_model(pipeline, X_test, y_test, class_names=None, model_name='classification_model'):
    """
    Evaluate a classification model with enhanced visualizations
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    class_names : list, optional
        List of class names
    model_name : str
        Name of the model for saving visualizations
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Create directory for evaluation visualizations
    os.makedirs('results/figures/model_evaluation/classification', exist_ok=True)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Get probability estimates if available
    y_prob = None
    if hasattr(pipeline, 'predict_proba'):
        try:
            y_prob = pipeline.predict_proba(X_test)
            has_probabilities = True
        except:
            has_probabilities = False
    else:
        has_probabilities = False
    
    # Calculate basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Classification Model Evaluation:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1 Score (weighted): {f1:.4f}")
    print("  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Enhanced confusion matrix visualization
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create two confusion matrices: one with counts, one with percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # If class names are provided, use them
    if class_names is not None:
        labels = class_names
    else:
        n_classes = len(np.unique(y_test))
        labels = [f'Class {i}' for i in range(n_classes)]
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Plot percentages
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=16)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    fig = plt.gcf()
    
    # Import save_figure from visualization module
    from visualization import save_figure
    save_figure(fig, f"model_evaluation/classification/confusion_matrix_{model_name}")
    
    # If probabilities are available, create ROC curve
    if has_probabilities:
        from sklearn.metrics import roc_curve, auc
        from itertools import cycle
        
        # Compute ROC curve and ROC area for each class
        plt.figure(figsize=(10, 8))
        
        n_classes = len(np.unique(y_test))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            # Convert to binary classification problem (one-vs-rest)
            y_test_bin = (y_test == i).astype(int)
            y_score = y_prob[:, i]
            
            fpr[i], tpr[i], _ = roc_curve(y_test_bin, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {labels[i]} (area = {roc_auc[i]:.2f})')
        
        # Plot the diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig = plt.gcf()
        save_figure(fig, f"model_evaluation/classification/roc_curve_{model_name}")
        
        # Compute micro-average and macro-average ROC curves
        plt.figure(figsize=(10, 8))
        
        # Micro-average
        y_test_bin = np.eye(n_classes)[y_test.astype(int)]  # One-hot encoding
        y_score = y_prob
        
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        # Macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        
        # Plot micro and macro average ROC curves
        plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC curve (area = {roc_auc_micro:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr_macro, tpr_macro, label=f'Macro-average ROC curve (area = {roc_auc_macro:.2f})',
                color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Micro-Average and Macro-Average ROC Curves', fontsize=16)
        plt.legend(loc="lower right")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig = plt.gcf()
        save_figure(fig, f"model_evaluation/classification/roc_curve_average_{model_name}")
    
    # Create precision-recall curves
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(10, 8))
    
    n_classes = len(np.unique(y_test))
    precision_class = dict()
    recall_class = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        # Convert to binary classification problem (one-vs-rest)
        y_test_bin = (y_test == i).astype(int)
        
        if has_probabilities:
            y_score = y_prob[:, i]
        else:
            # If no probabilities, use binary predictions
            y_score = (y_pred == i).astype(float)
        
        precision_class[i], recall_class[i], _ = precision_recall_curve(y_test_bin, y_score)
        avg_precision[i] = average_precision_score(y_test_bin, y_score)
    
    # Plot precision-recall curves for each class
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall_class[i], precision_class[i], color=color, lw=2,
                 label=f'Class {labels[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig = plt.gcf()
    save_figure(fig, f"model_evaluation/classification/precision_recall_{model_name}")
    
    # Return comprehensive evaluation metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'probabilities': y_prob if has_probabilities else None,
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
