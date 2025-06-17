# Machine Learning Model Setup Checklist

## 1. Data Understanding and Exploration

### □ Load and examine the dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Basic info
print(df.info())
print(df.describe())
print(df.head())
```

### □ Check data quality
```python
# Missing values
print(df.isnull().sum())

# Duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Data types
print(df.dtypes)
```

### □ Exploratory Data Analysis (EDA)
```python
# Distribution plots
df.hist(figsize=(12, 8))
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

## 2. Data Preprocessing

### □ Handle missing values
```python
from sklearn.impute import SimpleImputer

# For numerical features
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# For categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
```

### □ Handle outliers
```python
from scipy import stats

# Z-score method
z_scores = np.abs(stats.zscore(df[numerical_cols]))
df_no_outliers = df[(z_scores < 3).all(axis=1)]

# IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### □ Encode categorical variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Label encoding for ordinal variables
le = LabelEncoder()
df['ordinal_feature'] = le.fit_transform(df['ordinal_feature'])

# One-hot encoding for nominal variables
df_encoded = pd.get_dummies(df, columns=['categorical_feature'], drop_first=True)
```

## 3. Feature Engineering and Selection

### □ Create new features
```python
# Example: Creating interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']

# Example: Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 50, 75, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
```

### □ Feature scaling/normalization
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalization
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)
```

### □ Feature selection
```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
rf = RandomForestClassifier()
rfe = RFE(estimator=rf, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Feature importance from tree-based models
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

## 4. Data Splitting

### □ Split data into train, validation, and test sets
```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

## 5. Model Selection and Training

### □ Choose appropriate algorithms
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}
```

### □ Train baseline models
```python
from sklearn.metrics import accuracy_score, classification_report

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    
    print(f"{name}: {accuracy:.4f}")
```

## 6. Hyperparameter Tuning

### □ Grid search or random search
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search example
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

### □ Cross-validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# K-fold cross-validation
cv_scores = cross_val_score(
    best_model, X_train, y_train, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)

print(f"CV scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

## 7. Model Evaluation and Validation

### □ Evaluate on validation set
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Predictions
y_val_pred = best_model.predict(X_val)

# Metrics
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred, average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
```

### □ Confusion matrix and ROC curve
```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve (for binary classification)
if len(np.unique(y)) == 2:
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
```

### □ Learning curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curves')
plt.legend()
plt.grid()
plt.show()
```

## 8. Final Model Testing

### □ Test on holdout test set
```python
# Final evaluation on test set
y_test_pred = best_model.predict(X_test)

print("Final Test Set Results:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

print("\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred))
```

### □ Feature importance analysis
```python
# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    plt.show()
```

## 9. Model Interpretation and Documentation

### □ Model interpretability
```python
# SHAP values for model interpretation
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)
```

### □ Document model performance
```python
# Create performance summary
performance_summary = {
    'Model': type(best_model).__name__,
    'Best Parameters': grid_search.best_params_,
    'CV Score': grid_search.best_score_,
    'Test Accuracy': accuracy_score(y_test, y_test_pred),
    'Test Precision': precision_score(y_test, y_test_pred, average='weighted'),
    'Test Recall': recall_score(y_test, y_test_pred, average='weighted'),
    'Test F1': f1_score(y_test, y_test_pred, average='weighted')
}

print("Performance Summary:")
for key, value in performance_summary.items():
    print(f"{key}: {value}")
```

## 10. Model Deployment Preparation

### □ Save the trained model
```python
import joblib
import pickle

# Save with joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save with pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(best_model, f)
```

### □ Create prediction function
```python
def predict_new_data(new_data, model_path='best_model.pkl', scaler_path='scaler.pkl'):
    """
    Function to make predictions on new data
    """
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Preprocess new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    
    return predictions, probabilities

# Example usage
# predictions, probabilities = predict_new_data(new_sample)
```

## Checklist Summary

- [ ] Data exploration and understanding
- [ ] Data preprocessing and cleaning
- [ ] Feature engineering and selection
- [ ] Data splitting (train/val/test)
- [ ] Model selection and baseline training
- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Model evaluation on validation set
- [ ] Learning curve analysis
- [ ] Final testing on holdout set
- [ ] Model interpretation and documentation
- [ ] Model saving and deployment preparation

## Additional Considerations

### Performance Monitoring
- Set up monitoring for data drift
- Track model performance over time
- Plan for model retraining schedule

### Ethical Considerations
- Check for bias in predictions across different groups
- Ensure fairness and transparency
- Document limitations and assumptions

### Production Readiness
- Optimize for inference speed if needed
- Test scalability requirements
- Implement proper error handling
- Set up logging and monitoring