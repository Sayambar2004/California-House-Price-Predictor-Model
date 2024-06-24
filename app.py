from flask import Flask, render_template, request
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow all origins for testing purposes

# Load the model
model = pickle.load(open("Model.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form.get('Income'))
        house_age = float(request.form.get('age'))
        rooms = float(request.form.get('rooms'))
        bedrooms = float(request.form.get('bedrooms'))
        occupancy = float(request.form.get('occupancy'))
        population = float(request.form.get('Population'))
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))

        input_data = np.array([[income, house_age, rooms, bedrooms, occupancy, population, latitude, longitude]])
        prediction = model.predict(input_data)[0]

        formatted_prediction = f"{prediction:.2f}"

        return formatted_prediction
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
