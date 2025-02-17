from flask import Flask, render_template, request

import pickle

app = Flask(__name__)

# Load the trained model (replace 'heart_disease_model.pkl' with your model file)
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        MedInc = int(request.form['MedInc'])
        HouseAge = int(request.form['HouseAge'])
        AveRooms = int(request.form['AveRooms'])
        AveBedrms = int(request.form['AveBedrms'])
        Population = int(request.form['Population'])
        AveOccup = int(request.form['AveOccup'])
        Latitude = int(request.form['Latitude'])
        Longitude = int(request.form['Longitude'])
        

        # Create input data for prediction
        input_data = [[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]]

        # Make prediction using the trained model
        price = model.predict(input_data)[0]

        # Return the prediction result
        result = price
        return render_template('result.html', prediction=result)

if __name__== "__main__":
    app.run(debug=True)