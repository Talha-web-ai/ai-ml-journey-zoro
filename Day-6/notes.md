# 🧠 Day 6: SHAP Explainability + Bias/Fairness Testing
**Project:** Titanic Survival Prediction  
**Model:** XGBoostClassifier  
**Goal:** Understand model behavior using SHAP + Check for bias in predictions.

---

## 🔍 1. SHAP Explainability

### ✅ What is SHAP?
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain individual predictions. It tells us:
- Which features push a prediction **higher** (toward survival)
- Which features push it **lower** (toward not surviving)

### 📊 Visuals Used:
- `summary_plot`: Global feature importance (based on SHAP values)
- `force_plot`: Shows how each feature affected a single prediction

### 💡 Key SHAP Insights:
- **Sex (male)** strongly decreases survival prediction
- **Fare** and **Pclass** have positive impact on survival
- **Age** and **SibSp** contribute variably depending on the case

---

## ⚖️ 2. Bias & Fairness Testing

We tested if the model is biased toward certain groups by comparing **prediction accuracy** across:

### ✅ Gender:
| Group  | Accuracy |
|--------|----------|
| Female | 0.78     |
| Male   | 0.81     |

➡️ Small 3% difference. Acceptable, but model performs slightly better for males.

---

### ✅ Passenger Class:
| Pclass | Accuracy |
|--------|----------|
| 1st    | 0.81     |
| 2nd    | 0.85     |
| 3rd    | 0.77     |

➡️ Larger 8% difference. Model performs better for higher classes (wealthier passengers), indicating potential bias.

---

## ✅ Final Summary:

- Trained and explained a working ML model (XGBoost)
- Visualized global & individual predictions using SHAP
- Validated fairness across Gender and Class groups
- Prepared model for deployment in the next step

---

## 🚀 Next Up: Day 7 – Deploying the Model with Flask API

