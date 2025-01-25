from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained models and encoders
encoder = pickle.load(open('encoder_1.pkl', 'rb'))
scaler = pickle.load(open('scaler_1.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmax_1.pkl', 'rb'))
model = pickle.load(open('final_model_rf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input values from the form
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])
            humidity = float(request.form['humidity'])
            wind_kph = float(request.form['wind_kph'])
            pressure_mb = float(request.form['pressure_mb'])
            precip_mm = float(request.form['precip_mm'])
            wind_direction = request.form['wind_direction']

            # Scale each feature individually for StandardScaler
            latitude_scaled = scaler.transform([[latitude]])[0][0]
            longitude_scaled = scaler.transform([[longitude]])[0][0]
            pressure_scaled = scaler.transform([[pressure_mb]])[0][0]

            # Scale each feature individually for MinMaxScaler
            humidity_scaled = minmax_scaler.transform([[humidity]])[0][0]
            wind_kph_scaled = minmax_scaler.transform([[wind_kph]])[0][0]
            precip_mm_scaled = minmax_scaler.transform([[precip_mm]])[0][0]

            # Combine scaled values into a single array
            standard_scaled = np.array([latitude_scaled, longitude_scaled, pressure_scaled])
            minmax_scaled = np.array([humidity_scaled, wind_kph_scaled, precip_mm_scaled])

            # One-hot encode wind direction (directly returns a dense array)
            wind_encoded = encoder.transform([[wind_direction]])

            # Flatten the encoded array and concatenate
            inputs = np.concatenate([standard_scaled, minmax_scaled, wind_encoded.flatten()])

            # Predict temperature
            inputs = inputs.reshape(1, -1)  # Reshape for the model
            predicted_temp = model.predict(inputs)

            return render_template('result.html', prediction=round(predicted_temp[0], 2))

        except Exception as e:
            return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
