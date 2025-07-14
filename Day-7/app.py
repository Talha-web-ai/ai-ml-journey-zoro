from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = xgb.XGBClassifier()
model.load_model("model/xgboost_titanic_model.json")

# Define input features expected
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

@app.route('/')
def home():
    return "🚀 Titanic Survival Prediction API is Live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = df[FEATURES]  # enforce column order
        prediction = model.predict(df)[0]
        return jsonify({
            "prediction": int(prediction),
            "survived": bool(prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
