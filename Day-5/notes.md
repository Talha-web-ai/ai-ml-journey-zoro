
# 🧠 Day 5: XGBoost with Hyperparameter Tuning on Titanic Dataset

---

## 🔹 1. Dataset Used
- **Name**: Titanic Dataset  
- **Source**: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv  
- **Task**: Predict survival (binary classification)

---

## 🔹 2. Preprocessing Steps
```python
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
```
- Handled missing values
- Dropped irrelevant columns
- No encoding needed (some columns were already numeric)

---

## 🔹 3. Feature/Target Split
```python
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🔹 4. Model Used: XGBoost
```python
xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)
```
- `objective='binary:logistic'` → binary classification
- Other params optimized using `RandomizedSearchCV`

---

## 🔹 5. Hyperparameter Tuning with RandomizedSearchCV
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=1
)
random_search.fit(X_train, y_train)
```

✅ **Best Params Output**:
```bash
{'subsample': 0.8, 'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.01, 'colsample_bytree': 0.8}
```

---

## 🔹 6. Model Evaluation
```python
y_pred = best_model.predict(X_test)
```

**Confusion Matrix**:
```
[[91 14]
 [22 52]]
```

**Accuracy Score**:
```
0.7988
```

**Classification Report**:
- Class 0 (Not Survived): Precision 0.81, Recall 0.87
- Class 1 (Survived): Precision 0.79, Recall 0.70

---

## 🔹 7. Feature Importance
```python
import matplotlib.pyplot as plt
import seaborn as sns

importances = best_model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("XGBoost Feature Importance")
plt.show()
```

---

## 🧠 Key Concepts Covered
- `XGBClassifier`
- `RandomizedSearchCV`: faster hyperparameter tuning
- Grid vs Random search
- Evaluation metrics
- Feature importance
