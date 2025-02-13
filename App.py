import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Set UI
st.set_page_config(page_title="SkySaver", layout="centered")
st.title("SkySaver ✈️")
st.markdown(
"""
    <style>
        .stApp {
            background-color: #ADD8E6;;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Use EXACT feature names from the trained model
airlines = ['Air_India','AirAsia', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara']
source_cities = ['Bangalore','Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
destination_cities = ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
departure_times = ['Afternoon','Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']
arrival_times = ['Afternoon','Early_Morning', 'Evening', 'Late_Night', 'Morning', 'Night']
stops_options = ['zero','one', 'two_or_more']  # Matches training data
classes = ['Economy','Business']

# Collect user input
airline = st.selectbox("Select Airline", airlines)
source_city = st.selectbox("Source City", source_cities)
destination_city = st.selectbox("Destination City", destination_cities)
departure_time = st.selectbox("Departure Time", departure_times)
arrival_time = st.selectbox("Arrival Time", arrival_times)
stops = st.selectbox("Number of Stops", stops_options)
travel_class = st.selectbox("Class", classes)
duration = st.number_input("Flight Duration (in minutes)", min_value=30, max_value=1000, step=10)
days_left = st.number_input("Days Left for Travel", min_value=1, max_value=365, step=1)

# Create input dictionary
input_data = {"duration": duration, "days_left": days_left}

# One-hot encoding to match training feature names
for col in airlines:
    input_data[f"airline_{col}"] = 1 if airline == col else 0

for col in source_cities:
    input_data[f"source_city_{col}"] = 1 if source_city == col else 0

for col in destination_cities:
    input_data[f"destination_city_{col}"] = 1 if destination_city == col else 0

for col in departure_times:
    input_data[f"departure_time_{col}"] = 1 if departure_time == col else 0

for col in arrival_times:
    input_data[f"arrival_time_{col}"] = 1 if arrival_time == col else 0

for col in stops_options:
    input_data[f"stops_{col}"] = 1 if stops == col else 0

for col in classes:
    input_data[f"class_{col}"] = 1 if travel_class == col else 0

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize numerical values
input_df[['duration', 'days_left']] = scaler.transform(input_df[['duration', 'days_left']])

# Ensure column order matches model training
model_features = ['duration', 'days_left', 'airline_AirAsia',
       'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo',
       'airline_SpiceJet', 'airline_Vistara', 'source_city_Bangalore',
       'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad',
       'source_city_Kolkata', 'source_city_Mumbai', 'departure_time_Afternoon',
       'departure_time_Early_Morning', 'departure_time_Evening',
       'departure_time_Late_Night', 'departure_time_Morning',
       'departure_time_Night', 'stops_one', 'stops_two_or_more', 'stops_zero',
       'arrival_time_Afternoon', 'arrival_time_Early_Morning',
       'arrival_time_Evening', 'arrival_time_Late_Night',
       'arrival_time_Morning', 'arrival_time_Night',
       'destination_city_Bangalore', 'destination_city_Chennai',
       'destination_city_Delhi', 'destination_city_Hyderabad',
       'destination_city_Kolkata', 'destination_city_Mumbai', 'class_Business',
       'class_Economy']

# Reorder columns to match training data
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Predict price
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Ticket Price: ₹{round(prediction, 2)}")
