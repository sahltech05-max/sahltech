from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('loan_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    result = 'Loan Approved' if prediction == 1 else 'Loan Rejected'
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)