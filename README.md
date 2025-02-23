# Data Preparation to Modelling

This repository demonstrates the complete workflow of data preparation and modelling, using a dataset from the Dodoma Region Industrial Dataset. The workflow includes data loading, feature engineering, model building, and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Data Loading](#data-loading)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [Way Forward](#way-forward)

## Introduction

This project aims to predict the "Employment" variable using various machine learning models. The steps involved in the workflow include:

1. Data Loading
2. Feature Engineering
3. Model Building
4. Model Evaluation
5. Conclusion
6. Way Forward

## Data Loading

The dataset used in this project is the Dodoma Region Industrial Dataset. The dataset is loaded and the selected features are defined.

```python name=data_loading.py
import pandas as pd

# Load the dataset with selected features
df = pd.read_csv('drive/MyDrive/Data Analysis with Python/DODOMA_REGION_INDUSTRIAL_DATASET_SELECTED_FEATURES.csv')

# Define the selected features
selected_features = ['Size_Class_Description', 'id', 'Street', 'Establishment_Name',
                     'Tel_No', 'Ward_Name', 'Mobile_No', 'ISICLev4', 'ISICLev4_Description',
                     'CPC_Description', 'CPC_Code', 'Year_Started', 'Postal_Address',
                     'District_Name', 'Manufacturing_Classification', 'Ownership_Description',
                     'Subsector', 'Region_Name', 'Zone_Name']

# Specify 'Employment' as the target column
target_column = 'Employment'

# Separate features and target
X = df[selected_features]
y = df[target_column]
```

## Feature Engineering

Feature engineering involves transforming raw data into meaningful features that can be used to improve model performance. In this project, the steps include handling missing values, encoding categorical variables, and scaling numerical features.

```python name=feature_engineering.py
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Define preprocessing for numerical columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)
```

## Model Building

The preprocessed data is used to build and train multiple machine learning models, including Random Forest, XGBoost, and Linear Regression.

```python name=model_building.py
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression()
}
```

## Model Evaluation

The models are evaluated using metrics such as Mean Squared Error (MSE) and R² score. Feature importance is also analysed for interpretation.

```python name=model_evaluation.py
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Train and evaluate the models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_preprocessed, y, cv=5, n_jobs=-1)  # Parallelize cross-validation
    results[name] = {
        'Mean Squared Error': mse,
        'R^2 Score': r2,
        'Cross-Validation Score': np.mean(cv_scores)
    }
    print(f"{name} completed.")

# Print the results
for name, metrics in results.items():
    print(f"Model: {name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")

# Plot feature importances for XGBoost
model = models['XGBoost']
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), X.columns[indices], rotation=90)
plt.xlim([-1, len(indices)])
plt.title('Feature Importances (XGBoost)')
plt.show()
```

## Conclusion

The XGBoost model demonstrated the best performance with an MSE of $3.6494$, an R² score of $0.8928$, and a cross-validation score of $0.8673$. These metrics indicate that the model effectively explains the variance in the target variable "Employment" and generalises well to unseen data.

## Way Forward

1. **Feature Analysis and Interpretation**
   - Analyse the feature importance scores to identify the key drivers of employment in the dataset.
   - Use these insights to inform decision-making and guide further feature engineering efforts.

2. **Model Deployment**
   - Deploy the XGBoost model in a production environment to make predictions on new data.
   - Set up monitoring to track the model's performance over time and ensure it continues to perform well on new data.

3. **Continuous Improvement**
   - Regularly update the model with new data to keep it up-to-date and improve its accuracy.
   - Experiment with hyperparameter tuning and other advanced techniques to further enhance the model's performance.

4. **Documentation and Reporting**
   - Document the model-building process, including data preprocessing, feature selection, and model evaluation.
   - Create reports and visualisations to communicate the model's performance and key findings to stakeholders.

5. **Consider Ensemble Methods:**
   - Explore the possibility of combining the XGBoost model with other models (e.g., Random Forest) to create an ensemble model that might improve overall performance.

By following these steps, the XGBoost model can be effectively utilised for predicting employment and continuously improved to maintain high performance.
