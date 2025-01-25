import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load pre-trained models and encoders
encoder = pickle.load(open('encoder_1.pkl', 'rb'))
scaler = pickle.load(open('scaler_1.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmax_1.pkl', 'rb'))
model = pickle.load(open('final_model_rf.pkl', 'rb'))

# Streamlit app interface
st.title('Temperature Prediction App')

st.write("Enter weather data to predict the temperature:")

# Input fields for the user
latitude = st.number_input('Latitude', value=0.0)
longitude = st.number_input('Longitude', value=0.0)
humidity = st.number_input('Humidity (%)', value=50.0)
wind_kph = st.number_input('Wind Speed (kph)', value=0.0)
pressure_mb = st.number_input('Pressure (mb)', value=1013.0)
precip_mm = st.number_input('Precipitation (mm)', value=0.0)

# Dropdown for wind direction (one-hot encoded)
wind_direction = st.selectbox('Wind Direction', ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW'])

# Button to predict temperature
if st.button('Predict Temperature'):
    try:
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

        # One-hot encode wind direction
        wind_encoded = encoder.transform([[wind_direction]])

        # Flatten the encoded array and concatenate
        inputs = np.concatenate([standard_scaled, minmax_scaled, wind_encoded.flatten()])

        # Predict temperature
        inputs = inputs.reshape(1, -1)  # Reshape for the model
        predicted_temp = model.predict(inputs)

        # Display the result
        st.write(f"Predicted Temperature: {round(predicted_temp[0], 2)} °C")

    except Exception as e:
        st.error(f"Error: {str(e)}")

