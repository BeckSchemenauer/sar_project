import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv')

X = df.iloc[:, 1:]
y = df['diseases']

class_counts = y.value_counts()
print("Unique classes and their counts:")
print(class_counts)

# keep only classes with more than 100 cases
y = y[y.isin(class_counts[class_counts > 100].index)]
X = X.loc[y.index]

print(f"Number of unique classes (after dropping): {y.nunique()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt',]
}

# Perform grid search
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Save best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_random_forest_model.pkl')

y_pred = best_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(report)

# Save results to file
with open('model_results.txt', 'w') as f:
    f.write(f'Best Parameters: {grid_search.best_params_}\n')
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(report)