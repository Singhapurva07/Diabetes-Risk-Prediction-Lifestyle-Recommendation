import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Create directories
for dir_name in ['static', 'templates']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

print("Loading and preprocessing data...")

# Load dataset
try:
    data = pd.read_csv('diabetes.csv')
    print(f"Dataset loaded successfully. Shape: {data.shape}")
    print(f"Dataset columns: {list(data.columns)}")
    print(f"Target distribution:\n{data['Outcome'].value_counts()}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    print("Please ensure 'diabetes.csv' is in the current directory")
    exit(1)

# Data cleaning: Handle zero values and outliers
print("\nCleaning data...")
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_columns:
    if column in data.columns:
        median_val = data[data[column] > 0][column].median()
        data[column] = data[column].replace(0, median_val)
        print(f"Replaced zeros in {column} with median: {median_val:.2f}")

# Cap outliers using IQR method
for column in data.columns[:-1]:  # Exclude Outcome
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_before = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    if outliers_before > 0:
        print(f"Capped {outliers_before} outliers in {column}")

# Feature engineering
print("\nCreating new features...")
data['Glucose_BMI'] = data['Glucose'] * data['BMI']
data['Age_DiabetesPedigree'] = data['Age'] * data['DiabetesPedigreeFunction']
data['Insulin_Glucose_Ratio'] = data['Insulin'] / (data['Glucose'] + 1e-6)
data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
data['BMI_Category'] = data['BMI_Category'].astype(float)
data['Glucose_Age'] = data['Glucose'] * data['Age']

print(f"Final dataset shape: {data.shape}")
print(f"New features created: {list(data.columns[-5:])}")

# Prepare features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Feature selection with RFE
print("\nPerforming feature selection...")
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=8)
rfe.fit(X_train_scaled, y_train)
selected_features = X_train_scaled.columns[rfe.support_].tolist()
print(f"Selected Features: {selected_features}")

# Filter data to selected features
X_train_selected = X_train_scaled[selected_features]
X_test_selected = X_test_scaled[selected_features]

# Handle class imbalance with SMOTE
print("\nApplying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
print(f"Original class distribution: {np.bincount(y_train)}")
print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")

# Convert back to DataFrame
X_train_balanced = pd.DataFrame(X_train_balanced, columns=selected_features)

# Define models with optimized parameters
print("\nTraining models...")
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        class_weight={0: 1.0, 1: 2.0}
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=2.0,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        class_weight={0: 1.0, 1: 2.0},
        verbose=-1
    )
}

# Train individual models
trained_models = {}
individual_scores = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    individual_scores[name] = accuracy
    trained_models[name] = model
    print(f"{name} Accuracy: {accuracy:.4f}")

# Create ensemble
print("\nCreating ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', trained_models['RandomForest']),
        ('xgb', trained_models['XGBoost']),
        ('lgbm', trained_models['LightGBM'])
    ],
    voting='soft'
)

# Train ensemble
ensemble.fit(X_train_balanced, y_train_balanced)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test_selected)
y_pred_proba = ensemble.predict_proba(X_test_selected)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
print(f"\nIndividual Model Accuracies:")
for name, score in individual_scores.items():
    print(f"{name}: {score:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_ensemble)
print(f"\nConfusion Matrix:")
print(cm)

# Feature importance visualization
print("\nCreating feature importance plot...")
try:
    importances = trained_models['RandomForest'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved to static/feature_importance.png")
except Exception as e:
    print(f"Error creating feature importance plot: {str(e)}")

# Save model components
print("\nSaving model components...")
try:
    joblib.dump(ensemble, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(selected_features, 'selected_features.pkl')
    print("Model, scaler, and features saved successfully!")
except Exception as e:
    print(f"Error saving model components: {str(e)}")

print(f"\n{'='*50}")
print("Training completed successfully!")
print("You can now run 'python app.py' to start the web application.")
print(f"{'='*50}")