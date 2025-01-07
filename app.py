from flask import Flask,render_template,url_for,request,jsonify,redirect
import pandas as pd
import numpy as np
import pickle
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

app= Flask(__name__)
label_encoders= pickle.load(open('label_encoders.pkl','rb'))
model= pickle.load(open('logistic_regression_model.pkl','rb'))
scalers=pickle.load(open('scalers.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()

    # Extract the data from the "data" key
    input_data = data['data']  # Accessing the nested 'data' dictionary

    # Extract each feature from the input data
    age = input_data['age']
    job = input_data['job']
    marital = input_data['marital']
    education = input_data['education']
    default = input_data['default']
    balance = input_data['balance']
    housing = input_data['housing']
    loan = input_data['loan']
    contact = input_data['contact']
    day = input_data['day']
    month = input_data['month']
    duration = input_data['duration']
    campaign = input_data['campaign']
    pdays = input_data['pdays']
    previous = input_data['previous']
    poutcome = input_data['poutcome']

    # Encode categorical variables using the label encoders
    job = label_encoders['job'].transform([job])[0]
    marital = label_encoders['marital'].transform([marital])[0]
    education = label_encoders['education'].transform([education])[0]
    default = label_encoders['default'].transform([default])[0]
    housing = label_encoders['housing'].transform([housing])[0]
    loan = label_encoders['loan'].transform([loan])[0]
    contact = label_encoders['contact'].transform([contact])[0]
    month = label_encoders['month'].transform([month])[0]
    poutcome = label_encoders['poutcome'].transform([poutcome])[0]

    # Scale numeric features using the scalers
    age = scalers['age'].transform([[age]])[0][0]
    balance = scalers['balance'].transform([[balance]])[0][0]
    day = scalers['day'].transform([[day]])[0][0]
    duration = scalers['duration'].transform([[duration]])[0][0]
    campaign = scalers['campaign'].transform([[campaign]])[0][0]
    pdays = scalers['pdays'].transform([[pdays]])[0][0]
    previous = scalers['previous'].transform([[previous]])[0][0]

    # Prepare the data for prediction
    input_data = np.array([[ 
        age, job, marital, education, default, balance, housing, loan,
        contact, day, month, duration, campaign, pdays, previous, poutcome
    ]])

    # Make prediction
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        return jsonify({
            'success': True,
            'message': 'The customer will not subscribe to the term deposit',
            'prediction': 0
        })
    else:
        return jsonify({
            'success': True,
            'message': 'The customer will subscribe to the term deposit',
            'prediction': 1
        })

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data from the HTML form submission
    age = request.form['age']
    job = request.form['job']
    marital = request.form['marital']
    education = request.form['education']
    default = request.form['default']
    balance = request.form['balance']
    housing = request.form['housing']
    loan = request.form['loan']
    contact = request.form['contact']
    day = request.form['day']
    month = request.form['month']
    duration = request.form['duration']
    campaign = request.form['campaign']
    pdays = request.form['pdays']
    previous = request.form['previous']
    poutcome = request.form['poutcome']

    # Encode categorical variables using the label encoders
    job = label_encoders['job'].transform([job])[0]
    marital = label_encoders['marital'].transform([marital])[0]
    education = label_encoders['education'].transform([education])[0]
    default = label_encoders['default'].transform([default])[0]
    housing = label_encoders['housing'].transform([housing])[0]
    loan = label_encoders['loan'].transform([loan])[0]
    contact = label_encoders['contact'].transform([contact])[0]
    month = label_encoders['month'].transform([month])[0]
    poutcome = label_encoders['poutcome'].transform([poutcome])[0]

    # Scale numeric features using the scalers
    age = scalers['age'].transform([[age]])[0][0]
    balance = scalers['balance'].transform([[balance]])[0][0]
    day = scalers['day'].transform([[day]])[0][0]
    duration = scalers['duration'].transform([[duration]])[0][0]
    campaign = scalers['campaign'].transform([[campaign]])[0][0]
    pdays = scalers['pdays'].transform([[pdays]])[0][0]
    previous = scalers['previous'].transform([[previous]])[0][0]

    # Prepare the data for prediction
    input_data = np.array([[ 
        age, job, marital, education, default, balance, housing, loan,
        contact, day, month, duration, campaign, pdays, previous, poutcome
    ]])

    # Make prediction
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        return render_template('index.html', prediction_text='The customer will not subscribe to the term deposit')
    elif prediction[0] == 1:
        return render_template('index.html', prediction_text='The customer will subscribe to the term deposit')
    else:
        return render_template('index.html', prediction_text='Output will be shown here')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)