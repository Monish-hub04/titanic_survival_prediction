# Titanic Survival Prediction Project

This project implements a complete Machine Learning pipeline to predict survival on the Titanic using the classic Kaggle dataset (accessed via Seaborn).

## Features
- **Exploratory Data Analysis (EDA)**: Visualizations of survival rates by Gender, Class, and Age.
- **Feature Engineering**: Creation of `FamilySize` and `IsAlone` features.
- **Data Preprocessing**: Handling missing values and categorical encoding.
- **Multiple Models**: Logistic Regression, Random Forest, and Decision Tree.
- **Evaluation**: Accuracy scores, Confusion Matrices, and Classification Reports.
- **Feature Importance**: Visualization of which features matter most for survival.

## Installation
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to see the EDA results, model evaluations, and a sample prediction:
```bash
python titanic_prediction.py
```

## Project Structure
- `titanic_prediction.py`: Main script containing the full ML pipeline.
- `requirements.txt`: List of required Python libraries.
- `eda_plots.png`: Visualizations from the EDA step.
- `feature_importance.png`: Feature importance plot from the Random Forest model.
