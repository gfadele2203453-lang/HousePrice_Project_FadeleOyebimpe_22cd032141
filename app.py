from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form['OverallQual']),
                float(request.form['GrLivArea']),
                float(request.form['TotalBsmtSF']),
                float(request.form['GarageCars']),
                float(request.form['FullBath']),
                float(request.form['YearBuilt'])]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)