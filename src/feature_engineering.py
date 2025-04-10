import pandas as pd
import numpy as np

def create_bmi_feature(data, weight_col=None, height_col=None):
    """
    Calculate BMI (Body Mass Index) as a new feature
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    weight_col : str, optional
        Name of weight column (in kilograms)
    height_col : str, optional
        Name of height column (in meters or centimeters)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with BMI feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Identify column names
    if weight_col is None:
        # Try to find the appropriate column name
        if 'Weight' in df.columns:
            weight_col = 'Weight'
        elif 'Weight (kg)' in df.columns:
            weight_col = 'Weight (kg)'
        else:
            # Use the first column with 'weight' in the name (case insensitive)
            weight_cols = [col for col in df.columns if 'weight' in col.lower()]
            if weight_cols:
                weight_col = weight_cols[0]
            else:
                raise KeyError("Could not find weight column in the dataset.")
    
    if height_col is None:
        # Try to find the appropriate column name
        if 'Height' in df.columns:
            height_col = 'Height'
        elif 'Height (cm)' in df.columns:
            height_col = 'Height (cm)'
            # Convert cm to meters for BMI calculation
            df['Height (m)'] = df[height_col] / 100
            height_col = 'Height (m)'
        else:
            # Use the first column with 'height' in the name (case insensitive)
            height_cols = [col for col in df.columns if 'height' in col.lower()]
            if height_cols:
                height_col = height_cols[0]
                # Check if it's in cm and convert if needed
                if 'cm' in height_col.lower():
                    df['Height (m)'] = df[height_col] / 100
                    height_col = 'Height (m)'
            else:
                raise KeyError("Could not find height column in the dataset.")
    
    print(f"Using columns for BMI calculation: {weight_col} / {height_col}")
    
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

def create_age_group_feature(data, age_col=None):
    """
    Create age group categories as a new feature
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    age_col : str, optional
        Name of age column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with age group feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Identify column name
    if age_col is None:
        # Try to find the appropriate column name
        if 'Age' in df.columns:
            age_col = 'Age'
        else:
            # Use the first column with 'age' in the name (case insensitive)
            age_cols = [col for col in df.columns if 'age' in col.lower()]
            if age_cols:
                age_col = age_cols[0]
            else:
                raise KeyError("Could not find age column in the dataset.")
    
    print(f"Using column for age groups: {age_col}")
    
    # Create age groups
    df['Age_Group'] = pd.cut(
        df[age_col],
        bins=[0, 18, 30, 40, 50, 60, float('inf')],
        labels=['Under 18', '18-30', '31-40', '41-50', '51-60', 'Over 60']
    )
    
    print(f"Created age group categories:")
    print(df['Age_Group'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    
    return df

def create_workout_intensity_feature(data, heart_rate_col=None, resting_hr_col=None):
    """
    Calculate workout intensity based on heart rate reserve
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    heart_rate_col : str, optional
        Name of workout heart rate column
    resting_hr_col : str, optional
        Name of resting heart rate column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with workout intensity feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Identify column names
    if heart_rate_col is None:
        # Try to find the appropriate column name
        if 'Heart_Rate' in df.columns:
            heart_rate_col = 'Heart_Rate'
        elif 'Heart Rate (bpm)' in df.columns:
            heart_rate_col = 'Heart Rate (bpm)'
        else:
            # Use the first column with 'heart rate' in the name (case insensitive)
            hr_cols = [col for col in df.columns if 'heart rate' in col.lower() and 'rest' not in col.lower()]
            if hr_cols:
                heart_rate_col = hr_cols[0]
            else:
                raise KeyError("Could not find heart rate column in the dataset.")
    
    if resting_hr_col is None:
        # Try to find the appropriate column name
        if 'Resting_Heart_Rate' in df.columns:
            resting_hr_col = 'Resting_Heart_Rate'
        elif 'Resting Heart Rate (bpm)' in df.columns:
            resting_hr_col = 'Resting Heart Rate (bpm)'
        else:
            # Use the first column with 'resting heart rate' in the name (case insensitive)
            rhr_cols = [col for col in df.columns if 'heart rate' in col.lower() and 'rest' in col.lower()]
            if rhr_cols:
                resting_hr_col = rhr_cols[0]
            else:
                raise KeyError("Could not find resting heart rate column in the dataset.")
    
    print(f"Using columns for intensity calculation: {heart_rate_col} and {resting_hr_col}")
    
    # Get age column
    age_col = None
    if 'Age' in df.columns:
        age_col = 'Age'
    else:
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
        else:
            raise KeyError("Could not find age column in the dataset.")
    
    # Calculate maximum heart rate (simple formula: 220 - age)
    df['Max_Heart_Rate'] = 220 - df[age_col]
    
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

def create_combined_lifestyle_score(data, sleep_col=None, water_col=None, calories_col=None):
    """
    Create a combined lifestyle score based on sleep, water intake, and calorie intake
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    sleep_col : str, optional
        Name of sleep hours column
    water_col : str, optional
        Name of water intake column
    calories_col : str, optional
        Name of daily calories column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with lifestyle score feature added
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Identify column names
    if sleep_col is None:
        # Try to find the appropriate column name
        if 'Sleep_Hours' in df.columns:
            sleep_col = 'Sleep_Hours'
        elif 'Sleep Hours' in df.columns:
            sleep_col = 'Sleep Hours'
        else:
            # Use the first column with 'sleep' in the name (case insensitive)
            sleep_cols = [col for col in df.columns if 'sleep' in col.lower()]
            if sleep_cols:
                sleep_col = sleep_cols[0]
            else:
                raise KeyError("Could not find sleep hours column in the dataset.")
    
    if water_col is None:
        # Try to find the appropriate column name
        if 'Water_Intake' in df.columns:
            water_col = 'Water_Intake'
        elif 'Water Intake (liters)' in df.columns:
            water_col = 'Water Intake (liters)'
        else:
            # Use the first column with 'water' in the name (case insensitive)
            water_cols = [col for col in df.columns if 'water' in col.lower()]
            if water_cols:
                water_col = water_cols[0]
            else:
                raise KeyError("Could not find water intake column in the dataset.")
    
    if calories_col is None:
        # Try to find the appropriate column name
        if 'Daily_Calories' in df.columns:
            calories_col = 'Daily_Calories'
        elif 'Daily Calories Intake' in df.columns:
            calories_col = 'Daily Calories Intake'
        else:
            # Use the first column with 'daily' and 'calorie' in the name (case insensitive)
            calorie_cols = [col for col in df.columns if 'daily' in col.lower() and 'calorie' in col.lower()]
            if calorie_cols:
                calories_col = calorie_cols[0]
            else:
                # Try just 'calorie' in the name
                calorie_cols = [col for col in df.columns if 'calorie' in col.lower() and 'burn' not in col.lower()]
                if calorie_cols:
                    calories_col = calorie_cols[0]
                else:
                    raise KeyError("Could not find daily calories column in the dataset.")
    
    print(f"Using columns for lifestyle score: {sleep_col}, {water_col}, and {calories_col}")
    
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
from src.preprocessing import create_workout_efficiency_category
