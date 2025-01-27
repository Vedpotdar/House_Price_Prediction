from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        parking = int(request.form['parking'])
        year_built = int(request.form['year_built'])
        distance = float(request.form['distance'])
        amenities = int(request.form['amenities'])
        location = int(request.form['location'])
        build_type = int(request.form['build_type'])
        furnish_status = int(request.form['furnish_status'])

        # Prepare input for the model
        input_data = np.array([area, bedrooms, bathrooms, parking,
                               year_built, distance, amenities,
                               location, build_type, furnish_status]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_price = round(float(prediction), 2)

        return render_template('index.html', price=predicted_price)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
