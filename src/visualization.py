import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set default style
sns.set(style="whitegrid")

def save_figure(fig, filename, fig_dir='results/figures', dpi=300):
    """
    Save a figure to disk
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename without extension
    fig_dir : str
        Directory to save figures
    dpi : int
        Dots per inch for the saved figure
    """
    # Create directory if it doesn't exist
    os.makedirs(fig_dir, exist_ok=True)
    
    # Save the figure
    fig_path = f"{fig_dir}/{filename}.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

def plot_correlation_matrix(data, figsize=(12, 10), mask_upper=True):
    """
    Plot correlation matrix for the numeric features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with numeric features
    figsize : tuple
        Figure size
    mask_upper : bool
        Whether to mask the upper triangle
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with correlation matrix plot
    """
    # Calculate correlation matrix
    corr = data.select_dtypes(include=['float64', 'int64']).corr()
    
    # Create mask for upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    if mask_upper:
        mask[np.triu_indices_from(mask, k=1)] = True
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate heatmap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, 
                annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax)
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    
    return fig

def plot_feature_importance(model, feature_names, n_top=15, figsize=(10, 8)):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : trained model
        Model with feature_importances_ attribute (e.g. RandomForest, XGBoost)
    feature_names : list
        List of feature names
    n_top : int
        Number of top features to show
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with feature importance plot
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        if hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients as importance
            if len(model.coef_.shape) > 1:
                # For multiclass models, average importances across classes
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    else:
        importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Take the top n features
    indices = indices[:n_top]
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    ax.barh(range(len(top_importances)), top_importances, align='center')
    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Feature Importance')
    
    # Invert y-axis to show most important feature at the top
    ax.invert_yaxis()
    
    return fig

def plot_categorical_distribution(data, column, figsize=(10, 6)):
    """
    Plot the distribution of a categorical variable
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset
    column : str
        Categorical column name
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with categorical distribution plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get value counts and normalize
    value_counts = data[column].value_counts().sort_values(ascending=False)
    value_counts_pct = data[column].value_counts(normalize=True).sort_values(ascending=False) * 100
    
    # Create bar plot
    bars = sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    
    # Add percentage labels
    for i, (count, pct) in enumerate(zip(value_counts, value_counts_pct)):
        ax.text(i, count, f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
    
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    
    # Rotate x labels if there are many categories
    if len(value_counts) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

def plot_numeric_features_distribution(data, columns=None, figsize=(16, 12)):
    """
    Plot distribution of numeric features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset
    columns : list, optional
        List of numeric columns to plot. If None, all numeric columns are used.
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with distribution plots
    """
    # Select numeric columns if not specified
    if columns is None:
        columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        # Filter to only include numeric columns from the provided list
        columns = [col for col in columns if col in data.select_dtypes(include=['int64', 'float64']).columns]
    
    # Calculate number of rows/columns for subplots
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each feature
    for i, col in enumerate(columns):
        if i < len(axes):
            # Create histogram with KDE
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            
            # Add descriptive statistics
            mean = data[col].mean()
            median = data[col].median()
            std = data[col].std()
            axes[i].axvline(mean, color='r', linestyle='-', label=f'Mean: {mean:.2f}')
            axes[i].axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
            axes[i].legend()
            
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    return fig

def plot_target_vs_feature(data, target, feature, figsize=(10, 6), categorical=False):
    """
    Plot relationship between a target variable and a feature
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset
    target : str
        Name of the target column
    feature : str
        Name of the feature column
    figsize : tuple
        Figure size
    categorical : bool
        Whether the feature is categorical
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with target vs feature plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if categorical:
        # Group by categorical feature and calculate mean of target
        grouped = data.groupby(feature)[target].mean().sort_values(ascending=False)
        
        # Create bar plot
        bars = sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
        
        ax.set_title(f'Mean {target} by {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel(f'Mean {target}')
        
        # Add value labels
        for i, val in enumerate(grouped.values):
            ax.text(i, val, f'{val:.2f}', ha='center', va='bottom')
            
        # Rotate x labels if there are many categories
        if len(grouped) > 5:
            plt.xticks(rotation=45, ha='right')
    else:
        # For numeric features, use scatter plot with regression line
        sns.regplot(x=feature, y=target, data=data, scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'}, ax=ax)
        
        # Calculate correlation
        corr = data[[feature, target]].corr().iloc[0, 1]
        ax.set_title(f'{target} vs {feature} (Correlation: {corr:.3f})')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
    
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(8, 6)):
    """
    Plot confusion matrix for classification results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        List of label names
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with confusion matrix plot
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    
    # Set labels
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    return fig
