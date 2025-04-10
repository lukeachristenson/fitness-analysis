# AI Proposal

**Course:** CS 4100: Artificial Intelligence (Fall 2025) – Practical Track

## 1. Team Members

- Luke Christenson  
- Ipek Ozcan

This project will be conducted as a two-person team. One team member will focus on data preprocessing and adjustment for machine learning analysis, while the other will focus on data visualization and interpretability. Both team members will contribute to analyzing the algorithms used.

## 2. Problem Description and Formulation

The goal of this project is to analyze the relationships between workout performance and various lifestyle and health factors using a Workout & Fitness Tracker dataset from Kaggle. Specifically, we aim to investigate how variables such as sleep hours, water intake, body fat percentage, resting heart rate, and daily calorie intake influence workout efficiency and performance metrics (e.g., calories burned and heart rate during exercise).

### Inputs

- **Demographic:** Age, Gender, Height, Weight  
- **Lifestyle Metrics:** Sleep Hours, Water Intake, Daily Calories Intake, Mood  
- **Health Metrics:** Resting Heart Rate, Body Fat (%), VO2 Max  
- **Activity Metrics:** Workout Type, Workout Duration, Heart Rate (bpm), Steps, Distance

### Outputs

We will explore two possible approaches:

- **Classification:** Predicting “Workout Efficiency” as a categorical target (Low, Medium, High).
- **Regression:** Predicting “Calories Burned” or another performance metric as a continuous value.

The results will provide insights into which factors contribute to workout efficiency, potentially informing recommendations for fitness optimization.

## 3. Ideal Outcome and Expected Findings

- **Data-Driven Insights:** Identify correlations between fitness performance and factors such as sleep and nutrition.  
- **Predictive Model:** Develop a machine learning model to estimate workout efficiency based on input metrics.  
- **Interpretability:** Use feature importance analysis to determine which factors are most influential in predicting workout success.

## 4. Methods and Algorithms

- **Data Preprocessing:**  
  - Handle missing values  
  - Normalize ranges  
  - Create derived features where applicable

- **Exploratory Data Analysis (EDA):**  
  - Generate correlation matrices  
  - Create visualizations  
  - Summarize statistical data

- **Machine Learning Models:**  
  - *Baseline:* Linear Regression, Logistic Regression  
  - *Further:* Random Forest, XGBoost, or simple Neural Networks  
  - Hyperparameter tuning for performance optimization

- **Evaluation Metrics:**  
  - Mean Squared Error (for regression)  
  - Accuracy/F1-score (for classification)

## 5. Tools, Libraries, and Required Learning

- **Libraries:**  
  - scikit-learn  
  - PyTorch  
  - Pandas  
  - Matplotlib

- **Platform:** Kaggle (for dataset hosting and potential competition submission)

- **Additional Learning:**  
  - Potential use of SHAP values for interpretability  
  - Optimization techniques for model performance

## 6. Dataset

The dataset is the **Workout & Fitness Tracker Dataset** from Kaggle, containing over 10,000 records of workout and health-related metrics. It includes demographic information, exercise performance data, and lifestyle habits, making it well-suited for analyzing fitness trends and building predictive models.

**Dataset URL:**  
[Workout & Fitness Tracker Dataset](https://www.kaggle.com/datasets/adilshamim8/workout-and-fitness-tracker-data)

## 7. Milestone and Timeline

### Halfway Milestone (April 11)

1. **Data Preparation:**  
   - Clean and preprocess dataset  
   - Handle missing values  
   - Normalize features

2. **Initial Analysis:**  
   - Conduct EDA to explore correlations and feature distributions

3. **Baseline Model:**  
   - Implement at least one simple predictive model and assess initial performance

### Week-by-Week Plan

- **Week 1 (Mar 12 – Mar 18):** Finalize project scope, set up environment, verify dataset quality.  
- **Week 2 (Mar 19 – Mar 25):** Complete data preprocessing and EDA.  
- **Week 3 (Mar 26 – Apr 1):** Implement baseline model and evaluate initial results.  
- **Week 4 (Apr 2 – Apr 8):** Experiment with advanced models and refine hyperparameters.  
- **Week 5 (Apr 9 – Apr 15):** Milestone checkpoint: present preliminary findings.  
- **Week 6 (Apr 16 – Apr 22):** Focus on model improvements, interpretability analysis, and visualization.  
- **Week 7 (Apr 23 – Apr 30):** Finalize results and complete the report.

## 8. Challenges and Mitigation Strategies

- **Data Quality Issues:** Handle missing or inconsistent values through imputation or filtering.
- **Overfitting:** Use proper train-validation-test splits and regularization techniques.
- **Scalability:** Ensure the approach remains efficient as data complexity increases.

## 9. Conclusion

This project will provide insights into the impact of lifestyle factors on workout performance using machine learning techniques. By systematically analyzing data, training predictive models, and interpreting results, we aim to deliver meaningful conclusions that could inform fitness strategies. This work aligns with AI applications in health analytics and presents an opportunity to apply real-world machine learning methodologies.

---

You now have a Markdown version of your AI proposal ready for further editing or submission.