import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats

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

def plot_correlation_matrix(data, figsize=(14, 12), mask_upper=True, min_periods=20, threshold=0.6):
    """
    Plot correlation matrix for the numeric features with improved handling of NaN values
    and focusing on significant correlations
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with numeric features
    figsize : tuple
        Figure size
    mask_upper : bool
        Whether to mask the upper triangle
    min_periods : int
        Minimum number of valid observations required for correlation calculation
    threshold : float
        Correlation strength threshold for highlighting (0 to 1)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with correlation matrix plot
    """
    # Calculate correlation matrix
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Remove columns with too many missing values (>50%)
    valid_threshold = len(numeric_data) * 0.5
    valid_columns = [col for col in numeric_data.columns if numeric_data[col].count() >= valid_threshold]
    numeric_data = numeric_data[valid_columns]
    
    print(f"Computing correlation matrix for {len(valid_columns)} numeric features")
    
    # Calculate correlation matrix with minimum observations requirement
    corr = numeric_data.corr(method='pearson', min_periods=min_periods)
    
    # Create a copy for highlighting significant correlations
    corr_significant = corr.copy()
    
    # Create a mask for non-significant correlations
    highlight_mask = np.abs(corr) < threshold
    
    # Fill NaN values with 0 for visualization
    corr = corr.fillna(0)
    
    # Create mask for upper triangle
    triangle_mask = np.zeros_like(corr, dtype=bool)
    if mask_upper:
        triangle_mask[np.triu_indices_from(triangle_mask, k=1)] = True
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]), gridspec_kw={'width_ratios': [1, 1]})
    
    # Generate heatmap with diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Plot full heatmap with triangle mask
    sns.heatmap(
        corr, mask=triangle_mask, cmap=cmap, vmax=1, vmin=-1, center=0, 
        annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax1,
        annot_kws={"size": 8}
    )
    ax1.set_title('Full Correlation Matrix', fontsize=16, pad=20)
    
    # Create a combined mask for significant correlations view
    combined_mask = triangle_mask.copy()
    combined_mask = np.logical_or(combined_mask, highlight_mask)
    
    # Plot significant correlations only
    sns.heatmap(
        corr, mask=combined_mask, cmap=cmap, vmax=1, vmin=-1, center=0, 
        annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax2,
        annot_kws={"size": 8}
    )
    ax2.set_title(f'Significant Correlations (|r| ≥ {threshold})', fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create table of top correlations
    plt.figure(figsize=(12, 8))
    
    # Get upper triangle of correlation matrix (excluding diagonal)
    mask = np.triu(np.ones_like(corr), k=1).astype(bool)
    corr_flat = corr.mask(~mask).stack().sort_values(ascending=False)
    
    # Get top positive and negative correlations
    top_positive = corr_flat[corr_flat > threshold].head(15)
    top_negative = corr_flat[corr_flat < -threshold].head(15)
    
    # Create dataframe for table
    top_corr_data = {
        'Feature 1': [],
        'Feature 2': [],
        'Correlation': []
    }
    
    # Add positive correlations
    for (feat1, feat2), corr_val in top_positive.items():
        top_corr_data['Feature 1'].append(feat1)
        top_corr_data['Feature 2'].append(feat2)
        top_corr_data['Correlation'].append(corr_val)
    
    # Add negative correlations
    for (feat1, feat2), corr_val in top_negative.items():
        top_corr_data['Feature 1'].append(feat1)
        top_corr_data['Feature 2'].append(feat2)
        top_corr_data['Correlation'].append(corr_val)
    
    # Create dataframe and sort
    top_corr_df = pd.DataFrame(top_corr_data)
    top_corr_df = top_corr_df.sort_values('Correlation', ascending=False)
    
    # Display as a table
    plt.axis('off')
    plt.table(
        cellText=top_corr_df.values.round(3),
        colLabels=top_corr_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2', '#f2f2f2', '#f2f2f2'],
        cellColours=np.where(top_corr_df['Correlation'].values.reshape(-1, 1) > 0, 'lightblue', 'lightpink'),
    )
    plt.title('Top Feature Correlations', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the top correlations table
    table_fig = plt.gcf()
    save_figure(table_fig, "correlations/top_correlations_table")
    
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
    Plot relationship between a target variable and a feature with enhanced visualizations
    
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
        # Group by categorical feature and calculate statistics for the target
        grouped_stats = data.groupby(feature)[target].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)
        
        # Create bar plot with error bars
        bars = sns.barplot(x=grouped_stats.index, y=grouped_stats['mean'], ax=ax, 
                           order=grouped_stats.index,
                           yerr=grouped_stats['std'] / np.sqrt(grouped_stats['count']))
        
        ax.set_title(f'Mean {target} by {feature}', fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel(f'Mean {target}', fontsize=12)
        
        # Add value labels with count
        for i, (idx, row) in enumerate(grouped_stats.iterrows()):
            ax.text(i, row['mean'], 
                   f"μ={row['mean']:.2f}\nn={int(row['count'])}", 
                   ha='center', va='bottom', fontsize=9)
            
        # Rotate x labels if there are many categories
        if len(grouped_stats) > 4:
            plt.xticks(rotation=45, ha='right')
            
        # Add grid for readability
        ax.grid(axis='y', alpha=0.3)
        
    else:
        # Create a more comprehensive plot with multiple elements
        # First create a scatter plot with regression line and confidence interval
        g = sns.regplot(x=feature, y=target, data=data, 
                       scatter_kws={'alpha': 0.4, 'color': 'blue'}, 
                       line_kws={'color': 'red', 'lw': 2}, 
                       ax=ax)
        
        # Calculate correlation statistics
        corr, p_value = data[[feature, target]].dropna().corr().iloc[0, 1], \
                         scipy.stats.pearsonr(data[feature].dropna(), data[target].dropna())[1]
                         
        # Add correlation information
        correlation_text = f"Correlation: {corr:.3f}"
        if p_value < 0.001:
            correlation_text += " (p<0.001)"
        elif p_value < 0.01:
            correlation_text += f" (p<0.01)"
        elif p_value < 0.05:
            correlation_text += f" (p<0.05)"
        else:
            correlation_text += f" (p={p_value:.3f})"
            
        # Create a nice statistics box in the corner
        stats_text = (
            f"Statistics for {feature}:\n"
            f"Mean: {data[feature].mean():.2f}\n"
            f"Median: {data[feature].median():.2f}\n"
            f"SD: {data[feature].std():.2f}\n\n"
            f"{correlation_text}"
        )
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Set comprehensive title and labels
        ax.set_title(f'Relationship between {target} and {feature}', fontsize=14)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel(target, fontsize=12)
        
        # Add grid for readability
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_feature_relationships_heatmap(data, features=None, figsize=(16, 14)):
    """
    Create a comprehensive heatmap of scatter plots showing relationships
    between multiple features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset with features
    features : list, optional
        List of features to include in the heatmap. If None, will select
        numeric features up to a maximum of 8 to keep the plot readable.
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with heatmap of scatter plots
    """
    # Select features if not specified
    if features is None:
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        # Limit to top 8 features to keep plots readable
        if len(numeric_data.columns) > 8:
            # Try to select diverse features
            corr = numeric_data.corr().abs()
            # Sum correlations and sort features by lowest sum (most independent)
            corr_sums = corr.sum().sort_values()
            # Take first 8 features (most independent)
            features = corr_sums.index[:8].tolist()
        else:
            features = numeric_data.columns.tolist()
            
    # Create a subset of the data with selected features
    scatter_data = data[features].copy()
    
    # Create pair grid for scatter plots
    g = sns.PairGrid(scatter_data, diag_sharey=False)
    
    # Add scatter plots with regression line to upper triangle
    g.map_upper(sns.regplot, scatter_kws={'alpha': 0.5, 's': 20}, line_kws={'color': 'red'})
    
    # Add correlation coefficient to lower triangle with heatmap color
    def corrfunc(x, y, **kwargs):
        # Calculate correlation
        corr_coef = np.corrcoef(x, y)[0, 1]
        # Set color based on correlation strength and direction
        color = plt.cm.RdBu_r(0.5 * (corr_coef + 1))
        # Add text with correlation value
        ax = plt.gca()
        ax.text(0.5, 0.5, f'{corr_coef:.2f}', 
                transform=ax.transAxes, ha='center', va='center', 
                fontsize=14, fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    g.map_lower(corrfunc)
    
    # Add histograms to diagonal
    g.map_diag(sns.histplot, kde=True)
    
    # Adjust layout and add title
    g.fig.suptitle('Feature Relationships Matrix', fontsize=16, y=1.02)
    g.fig.tight_layout()
    
    return g.fig

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
