import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv')

X = df.iloc[:, 1:]
y = df['diseases']

print(f"Number of unique classes: {y.nunique()}")

# keep  classes with more than 25 cases
class_counts = y.value_counts()
y = y[y.isin(class_counts[class_counts > 100].index)]
X = X.loc[y.index]

print(f"Number of unique classes (after dropping): {y.nunique()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=200,
                                  max_depth=20,
                                  min_samples_split=5,
                                  min_samples_leaf=2,
                                  max_features='sqrt',
                                  bootstrap=True,
                                  random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))
