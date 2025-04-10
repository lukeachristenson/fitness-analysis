import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def identify_column_types(data):
    """
    Identify numeric and categorical columns in the dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    tuple
        (numeric_cols, categorical_cols)
    """
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Identified {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
    return numeric_cols, categorical_cols

def handle_missing_values(data, numeric_strategy='mean', categorical_strategy='most_frequent'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe with missing values
    numeric_strategy : str
        Strategy for imputing numeric columns ('mean', 'median', 'constant')
    categorical_strategy : str
        Strategy for imputing categorical columns ('most_frequent', 'constant')
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with imputed values
    """
    from sklearn.impute import SimpleImputer
    
    # Check for missing values
    missing_data = data.isnull().sum()
    cols_with_missing = missing_data[missing_data > 0].index.tolist()
    
    if not cols_with_missing:
        print("No missing values found in the dataset.")
        return data
    
    print(f"Found {len(cols_with_missing)} columns with missing values")
    for col in cols_with_missing:
        print(f"  - {col}: {missing_data[col]} missing values")
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(data)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy=numeric_strategy), 
             [col for col in numeric_cols if col in cols_with_missing]),
            ('cat', SimpleImputer(strategy=categorical_strategy), 
             [col for col in categorical_cols if col in cols_with_missing])
        ],
        remainder='passthrough'
    )
    
    # Setup pipeline and transform
    impute_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Get the column names in the right order for later
    all_cols = []
    all_cols.extend([col for col in numeric_cols if col in cols_with_missing])
    all_cols.extend([col for col in categorical_cols if col in cols_with_missing])
    all_cols.extend([col for col in data.columns if col not in cols_with_missing])
    
    # Fit and transform
    imputed_data = pd.DataFrame(
        impute_pipeline.fit_transform(data),
        columns=all_cols
    )
    
    print("Missing values have been imputed.")
    return imputed_data

def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Create a preprocessing pipeline for numeric and categorical features
    
    Parameters:
    -----------
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def create_workout_efficiency_category(data, calories_col=None, duration_col=None):
    """
    Create a derived feature for workout efficiency (calories burned per minute)
    and categorize it into Low, Medium, and High
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    calories_col : str, optional
        Name of the calories burned column
    duration_col : str, optional
        Name of the workout duration column (in minutes)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with the additional workout efficiency features
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Identify column names for calories and duration
    if calories_col is None:
        # Try to find the appropriate column name
        if 'Calories_Burned' in df.columns:
            calories_col = 'Calories_Burned'
        elif 'Calories Burned' in df.columns:
            calories_col = 'Calories Burned'
        else:
            # Use the first column with 'calorie' in the name (case insensitive)
            calorie_cols = [col for col in df.columns if 'calorie' in col.lower()]
            if calorie_cols:
                calories_col = calorie_cols[0]
            else:
                raise KeyError("Could not find calories column in the dataset.")
    
    if duration_col is None:
        # Try to find the appropriate column name
        if 'Workout_Duration' in df.columns:
            duration_col = 'Workout_Duration'
        elif 'Workout Duration (mins)' in df.columns:
            duration_col = 'Workout Duration (mins)'
        else:
            # Use the first column with 'duration' in the name (case insensitive)
            duration_cols = [col for col in df.columns if 'duration' in col.lower()]
            if duration_cols:
                duration_col = duration_cols[0]
            else:
                raise KeyError("Could not find workout duration column in the dataset.")
    
    print(f"Using columns for efficiency calculation: {calories_col} / {duration_col}")
    
    # Calculate efficiency (calories per minute)
    df['Workout_Efficiency_Score'] = df[calories_col] / df[duration_col]
    
    # Create categories based on quantiles
    efficiency_thresholds = df['Workout_Efficiency_Score'].quantile([0.33, 0.66])
    
    # Apply the categorization
    df['Workout_Efficiency'] = pd.cut(
        df['Workout_Efficiency_Score'],
        bins=[-float('inf'), efficiency_thresholds.iloc[0], efficiency_thresholds.iloc[1], float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    
    print(f"Created workout efficiency categories with distribution:")
    print(df['Workout_Efficiency'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    
    return df

def encode_categorical_target(data, target_col='Workout_Efficiency'):
    """
    Encode a categorical target variable
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of the categorical target column
        
    Returns:
    --------
    tuple
        (dataframe with encoded target, label encoder object)
    """
    from sklearn.preprocessing import LabelEncoder
    
    df = data.copy()
    le = LabelEncoder()
    df[f"{target_col}_Encoded"] = le.fit_transform(df[target_col])
    
    # Map the encoded values to original categories
    mapping = {i: category for i, category in enumerate(le.classes_)}
    print(f"Target encoding mapping: {mapping}")
    
    return df, le
