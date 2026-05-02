import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set style for plots
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Data Loading
# ==========================================
# Loading the Titanic dataset from seaborn
# Note: If you have train.csv, you can use: df = pd.read_csv('train.csv')
print("--- Step 1: Loading Data ---")
df = sns.load_dataset('titanic')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================
print("--- Step 2: Exploratory Data Analysis (EDA) ---")
# Basic statistics
print("Basic Statistics:")
print(df.describe())
print("\nNull Value Counts:")
print(df.isnull().sum())

# Plot Survival Rates by Gender
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Gender')

# Plot Survival Rates by Class
plt.subplot(1, 3, 2)
sns.barplot(x='class', y='survived', data=df)
plt.title('Survival Rate by Class')

# Plot Survival Rates by Age (using a boxplot for distribution)
plt.subplot(1, 3, 3)
sns.histplot(data=df, x='age', hue='survived', multiple='stack', kde=True)
plt.title('Survival by Age Distribution')

plt.tight_layout()
plt.savefig('eda_plots.png')
print("\nEDA plots saved to 'eda_plots.png'")

# ==========================================
# 3. Feature Engineering
# ==========================================
print("\n--- Step 3: Feature Engineering ---")
# Create FamilySize feature from sibsp + parch + 1 (self)
df['FamilySize'] = df['sibsp'] + df['parch'] + 1

# Create IsAlone feature (1 if FamilySize is 1, else 0)
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print("Created 'FamilySize' and 'IsAlone' features.")

# ==========================================
# 4. Data Preprocessing
# ==========================================
print("\n--- Step 4: Data Preprocessing ---")

# Handle missing values
# Age: Fill with median
df['age'] = df['age'].fillna(df['age'].median())

# Embarked: Fill with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Cabin/Deck: Since it has many nulls, we'll fill with 'Unknown' or just drop it as requested
# The user asked to handle it, but then drop irrelevant columns (Cabin). 
# We'll drop 'deck' (seaborn's cabin equivalent) and others.
# Also dropping 'alone' as we create 'IsAlone' from scratch.
df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1, inplace=True)

# Drop irrelevant columns (Name, Ticket, Cabin are not in seaborn dataset or already dropped)
# In seaborn 'deck' is Cabin. Name and Ticket are not present in seaborn's default load.
# If they were: df.drop(['name', 'ticket'], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex']) # male=1, female=0
df['embarked'] = le.fit_transform(df['embarked'])

print("Handled missing values and encoded categorical columns.")
print("Final columns used for training:", df.columns.tolist())

# ==========================================
# 5. Model Training
# ==========================================
print("\n--- Step 5: Model Training ---")

# Define Features and Target
X = df.drop('survived', axis=1)
y = df['survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "cr": classification_report(y_test, y_pred)
    }
    print(f"Trained {name}")

# ==========================================
# 6. Evaluation
# ==========================================
print("\n--- Step 6: Evaluation ---")
for name, result in results.items():
    print(f"\n{'='*30}\nModel: {name}\n{'='*30}")
    print(f"Accuracy Score: {result['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(result['cm'])
    print("\nClassification Report:")
    print(result['cr'])

# ==========================================
# 7. Feature Importance (Random Forest)
# ==========================================
print("\n--- Step 7: Feature Importance ---")
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance (Random Forest)')
plt.savefig('feature_importance.png')
print("Feature importance plot saved to 'feature_importance.png'")

# ==========================================
# 8. Prediction for Custom Passenger
# ==========================================
print("\n--- Step 8: Sample Prediction ---")

# Define a custom passenger matching the trained columns
# Columns: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'FamilySize', 'IsAlone']
# Example: 3rd class, Female (0), 25 years old, 1 sibling, 0 parents, 7.25 fare, S (2) embark, 2 family, 0 alone
custom_data = {
    'pclass': [3],
    'sex': [0],
    'age': [25.0],
    'sibsp': [1],
    'parch': [0],
    'fare': [7.25],
    'embarked': [2],
    'FamilySize': [2],
    'IsAlone': [0]
}
custom_passenger = pd.DataFrame(custom_data)

# Reorder columns to match X
custom_passenger = custom_passenger[X.columns]

# Using Random Forest for the sample prediction
prediction = rf_model.predict(custom_passenger)
probability = rf_model.predict_proba(custom_passenger)[0][1]

status = "Survived" if prediction[0] == 1 else "Did Not Survive"
print(f"Custom Passenger Input: {custom_data}")
print(f"Prediction: {status} (Probability: {probability:.2f})")

