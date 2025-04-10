# Fitness Data Analysis

## Project Overview

This project analyzes the relationships between workout performance and various lifestyle and health factors using a Workout & Fitness Tracker dataset. The analysis investigates how variables such as sleep hours, water intake, body fat percentage, resting heart rate, and daily calorie intake influence workout efficiency and performance metrics.

## Key Features

- **Comprehensive Data Analysis**: Extensive exploratory data analysis with enhanced visualizations
- **Advanced Feature Engineering**: Creation of derived features like BMI, workout intensity, and composite lifestyle scores
- **Lifestyle Impact Analysis**: Detailed quantification of how sleep, hydration, nutrition affect workout performance
- **Interactive Visualizations**: Comprehensive visualizations including:
  - Enhanced correlation matrices with significance highlighting
  - Feature relationship heatmaps with statistical annotations
  - Lifestyle factor impact quantification with percentage improvements
  - Detailed quartile analysis for non-linear patterns
- **Dual Modeling Approach**:
  - Classification models to predict workout efficiency categories (Low, Medium, High)
  - Regression models to predict calories burned during workouts
- **Model Evaluation**: Comprehensive performance metrics with detailed visualizations (ROC curves, confusion matrices)
- **Interpretability**: Feature importance analysis to identify key factors affecting workout performance

## Project Structure

```
├── data/                      # Data directory
│   └── workout_fitness_tracker_data.csv  # Dataset file
├── notebooks/                 # Jupyter notebooks for interactive analysis
├── results/                   # Results directory
│   ├── figures/               # Visualization outputs
│   │   ├── correlations/      # Correlation analysis visualizations
│   │   ├── distributions/     # Feature distribution visualizations
│   │   ├── feature_engineering/ # Engineered feature visualizations
│   │   ├── feature_relationships/ # Feature relationship visualizations
│   │   ├── lifestyle_analysis/ # Lifestyle impact analysis visualizations
│   │   └── model_evaluation/  # Model performance visualizations
│   │       ├── classification/ # Classification model evaluation
│   │       └── regression/    # Regression model evaluation
│   ├── models/                # Saved trained models
│   ├── analysis_summary.md    # Summary of analysis results in markdown
│   └── analysis_summary.txt   # Plain text summary of analysis results
├── src/                       # Source code
│   ├── __init__.py            # Make src a Python package
│   ├── data_loader.py         # Functions for loading data
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── feature_engineering.py # Feature creation and transformations
│   ├── models.py              # Model training and evaluation
│   ├── visualization.py       # Data visualization utilities
│   ├── main.py                # Main execution pipeline
│   └── inference.py           # Utilities for model inference
├── final_report.md            # Comprehensive final report with findings
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

The analysis provides comprehensive insights into:

1. **Lifestyle Impacts**: Quantified effects of lifestyle factors on workout performance
   - Sleep quality (22% improvement with optimal sleep)
   - Hydration level (18% improvement with proper hydration)
   - Combined lifestyle quality (35% difference between poor and excellent lifestyles)

2. **Feature Relationships**: Detailed correlation analysis with statistical significance testing
   - Enhanced correlation matrices identifying strongest relationships
   - Feature relationship heatmaps revealing complex interactions
   - Non-linear pattern detection through quantile analysis

3. **Performance Predictors**: Identified key determinants of workout efficiency
   - Heart rate during exercise (strongest predictor)
   - VO2 Max (strong indicator of aerobic capacity)
   - Workout duration (significant impact on calorie burn)
   - Sleep hours (critical lifestyle factor)

4. **Predictive Models**: Highly accurate machine learning models
   - Classification model with >98% accuracy for workout efficiency categories
   - Detailed visualization of model performance with ROC curves, confusion matrices
   - Feature importance rankings for interpretable insights

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