# 🧠 Day 3 – Linear & Logistic Regression with sklearn

## 📊 Dataset Used:
- **Breast Cancer Dataset** from `sklearn.datasets`
- **Objective**: Predict if a tumor is **malignant** or **benign** (binary classification)

---

## 🧹 Step 1: Preprocessing

### ✅ Load the Data:
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

### ✅ Create DataFrame:
```python
import pandas as pd
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
```

### ✅ Train-Test Split:
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 📈 Model 1: Logistic Regression

### ✅ Train the model:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
```

### ✅ Predict:
```python
y_pred = model.predict(X_test)
```

### ✅ Metrics:
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## 📊 Model 2: Linear Regression (for learning purpose)

Even though this is classification data, linear regression is applied for comparison.

### ✅ Train Linear Regression:
```python
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
```

### ✅ Predict & Evaluate:
```python
y_pred_lin = lin_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("MAE:", mean_absolute_error(y_test, y_pred_lin))
print("R2 Score:", r2_score(y_test, y_pred_lin))
```

---

## 📌 Summary:

| Model                | Task Type       | Key Metric       | Result (approx) |
|---------------------|------------------|------------------|------------------|
| Logistic Regression | Classification   | Accuracy         | ~95%             |
| Linear Regression   | Regression       | R² Score         | ~0.65–0.70       |

---

## 📚 Learnings:
- Logistic Regression is best for binary classification.
- Always match your model with the **type of target variable** (continuous vs. categorical).
- Confusion matrix helps break down model decisions.
- R² tells how well regression line fits the actual data.