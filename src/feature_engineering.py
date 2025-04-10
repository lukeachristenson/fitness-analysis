import pandas as pd
import numpy as np

def create_bmi_feature(data, weight_col='Weight', height_col='Height'):
    """
    Calculate BMI (Body Mass Index) as a new feature
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    weight_col : str
        Name of weight column (in kilograms)
    height_col : str
        Name of height column (in meters)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with BMI feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Calculate BMI
    df['BMI'] = df[weight_col] / (df[height_col] ** 2)
    
    # Create BMI categories
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 34.9, float('inf')],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Extremely Obese']
    )
    
    print(f"Created BMI feature with categories:")
    print(df['BMI_Category'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    
    return df

def create_age_group_feature(data, age_col='Age'):
    """
    Create age group categories as a new feature
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    age_col : str
        Name of age column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with age group feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Create age groups
    df['Age_Group'] = pd.cut(
        df[age_col],
        bins=[0, 18, 30, 40, 50, 60, float('inf')],
        labels=['Under 18', '18-30', '31-40', '41-50', '51-60', 'Over 60']
    )
    
    print(f"Created age group categories:")
    print(df['Age_Group'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    
    return df

def create_workout_intensity_feature(data, heart_rate_col='Heart_Rate', resting_hr_col='Resting_Heart_Rate'):
    """
    Calculate workout intensity based on heart rate reserve
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    heart_rate_col : str
        Name of workout heart rate column
    resting_hr_col : str
        Name of resting heart rate column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with workout intensity feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Calculate maximum heart rate (simple formula: 220 - age)
    df['Max_Heart_Rate'] = 220 - df['Age']
    
    # Calculate heart rate reserve (HRR)
    df['Heart_Rate_Reserve'] = df['Max_Heart_Rate'] - df[resting_hr_col]
    
    # Calculate workout intensity percentage (Karvonen formula)
    df['Workout_Intensity_Pct'] = (df[heart_rate_col] - df[resting_hr_col]) / df['Heart_Rate_Reserve'] * 100
    
    # Create intensity categories
    df['Workout_Intensity'] = pd.cut(
        df['Workout_Intensity_Pct'],
        bins=[0, 50, 70, 85, 100, float('inf')],
        labels=['Very Light', 'Light', 'Moderate', 'Vigorous', 'Maximum']
    )
    
    print(f"Created workout intensity categories:")
    print(df['Workout_Intensity'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    
    return df

def create_time_based_features(data, date_col='Date'):
    """
    Create time-based features from date column
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    date_col : str
        Name of date column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with time-based features added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract time features
    df['Day_of_Week'] = df[date_col].dt.day_name()
    df['Month'] = df[date_col].dt.month_name()
    df['Year'] = df[date_col].dt.year
    df['Is_Weekend'] = df[date_col].dt.dayofweek >= 5  # 5 = Saturday, 6 = Sunday
    
    print(f"Created time-based features")
    print(f"Weekend workouts: {df['Is_Weekend'].mean():.1%}")
    
    return df

def create_combined_lifestyle_score(data, sleep_col='Sleep_Hours', water_col='Water_Intake', calories_col='Daily_Calories'):
    """
    Create a combined lifestyle score based on sleep, water intake, and calorie intake
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    sleep_col : str
        Name of sleep hours column
    water_col : str
        Name of water intake column
    calories_col : str
        Name of daily calories column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with lifestyle score feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Calculate Z-scores for each metric
    sleep_z = (df[sleep_col] - df[sleep_col].mean()) / df[sleep_col].std()
    water_z = (df[water_col] - df[water_col].mean()) / df[water_col].std()
    calories_z = (df[calories_col] - df[calories_col].mean()) / df[calories_col].std()
    
    # Create combined score (simple average of z-scores)
    df['Lifestyle_Score'] = (sleep_z + water_z + calories_z) / 3
    
    # Create categories
    df['Lifestyle_Quality'] = pd.qcut(
        df['Lifestyle_Score'],
        q=5,
        labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
    )
    
    print(f"Created lifestyle quality categories:")
    print(df['Lifestyle_Quality'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    
    return df

def create_all_features(data):
    """
    Apply all feature engineering functions to create a rich feature set
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with all engineered features added
    """
    # Apply all feature engineering functions in sequence
    df = data.copy()
    
    print("Applying all feature engineering transformations...")
    
    # Create basic derived features
    df = create_bmi_feature(df)
    df = create_age_group_feature(df)
    df = create_workout_intensity_feature(df)
    
    # Create time-based features if 'Date' column exists
    if 'Date' in df.columns:
        df = create_time_based_features(df)
    
    # Create lifestyle score
    df = create_combined_lifestyle_score(df)
    
    # Create workout efficiency categories
    df = create_workout_efficiency_category(df)
    
    print("Feature engineering complete!")
    print(f"Original dataset: {data.shape[1]} features")
    print(f"Enhanced dataset: {df.shape[1]} features")
    
    return df

# Import the function since we reference it
from preprocessing import create_workout_efficiency_category
