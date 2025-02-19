import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import itertools

# Load dataset
df = pd.read_csv('src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv')

# Separate features and target variable
X = df.iloc[:, 1:]
y = df['diseases']

# Print unique classes and their counts
class_counts = y.value_counts()
print(f"Number of unique classes: {y.nunique()}")

# Keep only classes with more than 100 cases
y = y[y.isin(class_counts[class_counts > 100].index)]
X = X.loc[y.index]

print(f"Number of unique classes (after dropping): {y.nunique()}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define parameter grid
param_grid = {
    'n_estimators': [200],
    'max_depth': [70],
    'min_samples_split': [8],
    'min_samples_leaf': [1],
    'bootstrap': [True],
    'max_features': ['sqrt'],
    'random_state': [42]
}

# Iterate through parameter combinations
best_model = None
best_score = 0
best_params = None

for params in (dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())):
    print(f"Training with parameters: {params}")
    rf_model = RandomForestClassifier(**params, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    if accuracy > best_score:
        best_score = accuracy
        best_model = rf_model
        best_params = params

# Save best model
joblib.dump(best_model, 'best_random_forest_model.pkl')

# Evaluate best model
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)

# Save results to file
with open('model_results.txt', 'w') as f:
    f.write(f'Best Parameters: {best_params}\n')
    f.write(f'Best Accuracy: {best_score:.4f}\n')
    f.write(report)

print("Best model parameters saved.")
