# 🧠 Day 4 Notes – Classification Algorithms (KNN, SVM, Decision Tree)

---

## 📌 Dataset Used
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
```

---

## 📂 Data Split & Scaling
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 1️⃣ K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### ✅ Metrics
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

> 📊 **Accuracy:** ~96%

---

## 2️⃣ Support Vector Machine (SVM)
```python
from sklearn.svm import SVC

svm = SVC(kernel='linear')  # Try 'rbf' too
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
```

> 📊 **Accuracy:** ~96%

---

## 3️⃣ Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="gini", random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
```

> 📊 **Accuracy:** Varies with `max_depth`, `criterion`, etc.

---

## 🔍 Key Learnings

- 🔄 **Scaling is important** for KNN and SVM (distance-based methods), but not needed for Decision Trees.
- 🧮 **Confusion Matrix** helps understand TP, FP, FN, TN.
- 🎯 Use **classification report** to analyze precision, recall, and F1-score.
- ⚖️ Trade-offs in model choice depend on interpretability vs performance.

---