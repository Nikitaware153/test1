from flask import Flask,render_template, request, jsonify
import numpy as np
import pandas as pd


import pickle

# Your code here
with open('lr_model.pkl', 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__, template_folder='templates')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    open_price = float(request.form['open'])
    low_price = float(request.form['low'])
    close_price = float(request.form['close'])
    adj_close_price = float(request.form['adj_close'])
    volume = float(request.form['volume'])

    # Create a NumPy array from the input data
    input_data = np.array([[open_price, low_price, close_price, adj_close_price, volume]])

    # Use the model to make a prediction
    prediction = model.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'high_prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
