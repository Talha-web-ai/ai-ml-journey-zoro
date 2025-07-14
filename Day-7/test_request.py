import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Pclass": 2,
    "Sex": 0,
    "Age": 28,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 30.0,
    "Embarked": 1
}

response = requests.post(url, json=data)
print("Response:", response.json())
