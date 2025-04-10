# Fitness Data Analysis

## Project Overview

This project analyzes the relationships between workout performance and various lifestyle and health factors using a Workout & Fitness Tracker dataset. The analysis investigates how variables such as sleep hours, water intake, body fat percentage, resting heart rate, and daily calorie intake influence workout efficiency and performance metrics.

## Key Features

- **Comprehensive Data Analysis**: Exploratory data analysis of fitness and lifestyle variables
- **Advanced Feature Engineering**: Creation of derived features like BMI, workout intensity, and lifestyle quality scores
- **Dual Modeling Approach**:
  - Classification models to predict workout efficiency categories (Low, Medium, High)
  - Regression models to predict calories burned during workouts
- **Model Comparison**: Evaluation of multiple machine learning algorithms (Linear/Logistic Regression, Random Forest, XGBoost)
- **Interpretability**: Feature importance analysis to identify key factors affecting workout performance

## Project Structure

```
├── data/                      # Data directory
│   └── workout_fitness_tracker_data.csv  # Dataset file
├── notebooks/                 # Jupyter notebooks for interactive analysis
├── results/                   # Results directory
│   ├── figures/               # Visualization outputs
│   └── models/                # Saved trained models
├── src/                       # Source code
│   ├── __init__.py            # Make src a Python package
│   ├── data_loader.py         # Functions for loading data
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── feature_engineering.py # Feature creation and transformations
│   ├── models.py              # Model training and evaluation
│   ├── visualization.py       # Data visualization utilities
│   ├── main.py                # Main execution pipeline
│   └── inference.py           # Utilities for model inference
├── requirements.txt           # Project dependencies
├── run_analysis.py            # Script to run the analysis pipeline
└── README.md                  # Project documentation
```

## Dataset

The dataset used is the **Workout & Fitness Tracker Dataset** from Kaggle, containing records of workout and health-related metrics including:

- **Demographic Information**: Age, Gender, Height, Weight
- **Lifestyle Metrics**: Sleep Hours, Water Intake, Daily Calories Intake, Mood
- **Health Metrics**: Resting Heart Rate, Body Fat Percentage, VO2 Max
- **Activity Data**: Workout Type, Duration, Heart Rate during exercise, Steps, Distance

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

### Installation

1. Clone this repository
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Analysis

To execute the complete analysis pipeline:

```bash
python run_analysis.py
```

This will:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Engineer new features
4. Train and evaluate classification and regression models
5. Save results and visualizations

## Results

The analysis provides insights into:

1. **Correlations**: How different lifestyle factors correlate with workout performance
2. **Key Predictors**: Which variables are most important in predicting workout efficiency
3. **Predictive Models**: Machine learning models that can estimate workout performance based on health and lifestyle inputs

## Future Work

- Implement personalized fitness recommendation system
- Incorporate time-series analysis for tracking progress over time
- Deploy models as a web application
- Integrate with fitness tracking devices for real-time prediction

## Contributors

- Luke Christenson
- Ipek Ozcan

## Acknowledgments

- Dataset provided by Kaggle: [Workout & Fitness Tracker Dataset](https://www.kaggle.com/datasets/adilshamim8/workout-and-fitness-tracker-data)