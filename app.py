


import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ppscore as pps
from sklearn.preprocessing import OneHotEncoder



import streamlit as st
import pandas as pd
import joblib

# Load the dataset and model
raw_df = pd.read_csv('data_file.csv')
clean_df = raw_df.dropna()
model = joblib.load('model.pkl')

def page_1():
    st.title('Fare Predictor')

    st.info(
        f"#### **Business Requirement**: Predict Fare\n\n"
        f"* The client is interested in predicting fare based on several features.\n"
        f"* A machine learning model was built with the following features:\n"
        f"  * Year, Quarter, Airport1, Airport2, Distance (nsmiles), Passengers, etc."
    )

    # Define the categorical features and their options
    airport_options = ['EFD', 'ABQ', 'COS', 'DAL', 'DFW', 'PIT', 'ALB', 'DEN', 'ATL', 'AUS', 'PHX', 'BDL', 'SEA', 
                       'BHM', 'ELP', 'CAK', 'CLE', 'BNA', 'BOI', 'BOS', 'MHT', 'PVD', 'BTV', 'BUF', 'MDW', 'ORD', 
                       'CHS', 'CLT', 'CMH', 'STL', 'MYR', 'JAX', 'DTW', 'DSM', 'HOU', 'IAH', 'MCO', 'EUG', 'MSP', 
                       'EWR', 'HPN', 'ISP', 'JFK', 'LGA', 'SWF', 'RSW', 'GSP', 'GRR', 'GSO', 'LAS', 'IND', 'OAK', 
                       'SFO', 'SJC', 'FLL', 'MIA', 'BUR', 'LAX', 'LGB', 'ONT', 'SNA', 'LIT', 'SDF', 'CVG', 'MCI', 
                       'SAT', 'MEM', 'OMA', 'MKE', 'MSN', 'MSY', 'SAN', 'ORF', 'PHF', 'OKC', 'PDX', 'PHL', 'RDU', 
                       'RNO', 'ROC', 'SLC', 'TYS', 'CAE', 'XNA', 'CHI', 'DAY', 'NYC', 'Others']
    
    carrier_lg_options = ['G4', 'DL', 'WN', 'AA', 'UA', 'B6', 'AS', 'F9', 'NK', 'Others', 'MX', 'US', 'HP', 'CO', 
                          'YX', 'FL', 'NW', 'TW', 'RU', 'DH', 'TZ', 'JI', 'QQ', 'VX']
    
    carrier_low_options = ['G4', 'UA', 'WN', 'AA', 'B6', 'DL', 'F9', 'NK', 'AS', 'SY', 'Others', 'MX', 'US', 'NW', 
                           'CO', 'HP', 'FL', 'YX', 'NJ', 'TW', 'RU', 'DH', 'J7', 'JI', 'TZ', 'U5', 'VX', 'QQ', 'W7']

    st.subheader("Predict Fare")

    # Create input fields for user input in a single line
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input('Year', min_value=1993, max_value=2024, value=2018, step=1)
        quarter = st.selectbox('Quarter', [1, 2, 3, 4])
        nsmiles = st.number_input('Distance (miles)', min_value=0, value=550, step=1)
        
    with col2:
        airport_1 = st.selectbox('Departure Airport', airport_options)
        airport_2 = st.selectbox('Arrival Airport', airport_options)
        passengers = st.number_input('Number of Passengers', min_value=1, value=51, step=1)
        
    with col3:
        carrier_lg = st.selectbox('Large Carrier', carrier_lg_options)
        large_ms = st.number_input('Large Carrier Market Share (%)', min_value=0, max_value=100, value=52, step=1)
        fare_lg = st.number_input('Large Carrier Fare ($)', min_value=0, value=307, step=1)

 # Create another row for low-cost carrier inputs
    col4, col5, col6 = st.columns(3)
    with col4:
        carrier_low = st.selectbox('Low-Cost Carrier', carrier_low_options)
        
    with col5:
        lf_ms = st.number_input('Low-Cost Carrier Market Share (%)', min_value=0, max_value=100, value=45, step=1)

    with col6:
        fare_low = st.number_input('Low-Cost Carrier Fare ($)', min_value=0, value=280, step=1)

    if st.button('Run Predictive Analysis'):
        input_data = pd.DataFrame({
            'Year': [year], 'quarter': [quarter], 'airport_1': [airport_1], 'airport_2': [airport_2],
            'nsmiles': [nsmiles], 'passengers': [passengers], 'carrier_lg': [carrier_lg],
            'large_ms': [large_ms / 100], 'fare_lg': [fare_lg], 'carrier_low': [carrier_low],
            'lf_ms': [lf_ms / 100], 'fare_low': [fare_low]
        })
        
        categorical_features = ['airport_1', 'airport_2', 'carrier_lg', 'carrier_low']
        input_data_encoded = pd.get_dummies(input_data, columns=categorical_features)
        
        columns_used_in_training = model.feature_names_in_
        for col in columns_used_in_training:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0
                
        input_data_encoded = input_data_encoded[columns_used_in_training]
        prediction = model.predict(input_data_encoded)
        st.success(f'Predicted Fare: ${prediction[0]:.2f}')

# # Call the page_1 function to display the content
# page_1()