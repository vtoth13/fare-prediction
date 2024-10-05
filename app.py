


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
        


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df  = pd.read_csv("US Airline Flight Routes and Fares 1993-2024.csv")
# Global functions
@st.cache_data
def load_data():
    return pd.read_csv('data_file.csv')

@st.cache_data
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    df['Year'] = df['Year'].astype(int)
    df['quarter'] = df['quarter'].astype(int)
    return df

@st.cache_data
def engineer_features(df):
    df['profit_margin'] = (df['fare'] - df['fare_low']) / df['fare']
    return df

@st.cache_data
def one_hot_encode(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Changed 'sparse' to 'sparse_output'
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    return pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)


import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
# from ppscore import predictive_power_score
import ppscore as pps

# Utility functions
def dc_no_encoding_pipeline(df):
    numeric_columns = ['Year', 'quarter', 'nsmiles', 'passengers', 'large_ms', 'fare_lg', 'lf_ms', 'fare_low']
    pipeline = Pipeline([
        ("median_imputation", MeanMedianImputer(
            imputation_method="median",
            variables=numeric_columns)),
    ])

    clean_df = pipeline.fit_transform(df)
    return clean_df

def one_hot_encode(df):
    categorical_vars = ['airport_1', 'airport_2', 'carrier_lg', 'carrier_low']
    encoder = OneHotEncoder(
        variables=categorical_vars,
        drop_last=False)
    df_ohe = encoder.fit_transform(df)
    return df_ohe

# Main application
def load_data():
    # Replace this with your actual data loading logic
    return pd.read_csv("data_file.csv")

def clean_data(df):
    return dc_no_encoding_pipeline(df)

def engineer_features(df):
    # Add your feature engineering logic here
    df['price_per_mile'] = df['fare_lg'] / df['nsmiles']
    return df

def calculate_pps1(df, target):
    return pd.DataFrame(
        [
            {
                "x": x,
                "y": y,
                "ppscore": predictive_power_score(df[x], df[y]),
            }
            for x in df.columns
            for y in [target]
            if x != y
        ]
    )


def calculate_pps(df, target):
    # Get the Predictive Power Score (PPS) matrix for the entire DataFrame
    pps_matrix = pps.matrix(df)
    
    # Filter the PPS matrix to get only the scores for the specified target column
    # return pps_matrix[pps_matrix['target'] == target][['feature', 'ppscore']]
 
def page_2():
    import streamlit as st
    st.title('Fare Predictor')
    st.write("### Airline Fare Correlation Study")

    st.write(
        f"* [Business Requirement and Dataset](#business-requirement-1-data-visualisation-and-correlation-study)\n"
        f"* [Summary of Correlation Analysis](#summary-of-correlation-analysis)\n"
        f"* [Summary of PPS Analysis](#summary-of-pps-analysis)\n"
        f"* [Analysis of Most Correlated Features](#analysis-of-most-correlated-features)\n"
        f"* [Feature Relationships](#feature-relationships)\n"
        f"* [Conclusions](#conclusions)\n"
    )

    st.info(
        f"#### **Business Requirement 1**: Data Visualisation and Correlation Study\n\n"
        f"* We need to perform a correlation study to determine which features correlate most closely to the airline fare.\n"
        f"* A Pearson's correlation will indicate linear relationships between numerical variables.\n"
        f"* A Spearman's correlation will measure the monotonic relationships between variables.\n"
        f"* A Predictive Power Score (PPS) analysis will be performed to capture non-linear relationships.\n"
    )

    raw_df = load_data()
    clean_df = clean_data(raw_df)
    engineered_df = engineer_features(clean_df)

    if st.checkbox("Inspect airline fare dataset"):
        st.dataframe(raw_df.head(5))
        st.write(f"The dataset contains {raw_df.shape[0]} observations with {raw_df.shape[1]} attributes.")

    st.write("---")

    st.write(
        f"#### **Summary of Correlation Analysis**\n"
        f"* Correlations within the dataset were analysed using Spearman and Pearson correlations.\n"
        f"* For both correlations, all categorical features from the cleaned dataset were one hot encoded.\n"
    )

    def correlation(df, method):
        ohe_df = one_hot_encode(df)
        corr = ohe_df.corr(method=method)["fare_lg"].sort_values(
            key=abs, ascending=False)[1:].head(10)
        return corr

    if st.checkbox("View Pearson correlation results"):
        st.write(correlation(engineered_df, method="pearson"))
    if st.checkbox("View Spearman correlation results"):
        st.write(correlation(engineered_df, method="spearman"))

    st.write("---")

    st.write(
        f"#### **Summary of PPS Analysis**\n"
        f"* The PPS analysis provides insights into non-linear relationships between features and the target variable.\n"
        f"* It complements the linear correlation analysis by capturing more complex interactions.\n"
    )

    
    pps_scores = calculate_pps(engineered_df, "fare_lg")

    import streamlit as st
    import pandas as pd
    import plotly.express as px

# Load your data only once at the beginning of the app
    cleaned_df = pd.read_csv('data_file.csv')

# Create a checkbox for displaying the heatmap
    # if st.checkbox("View PPS heatmap"):
        
    # # Select numeric columns
    #     numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()

    #     corr_matrix = cleaned_df[numeric_cols].corr()

    # # Create a heatmap using Plotly
    #     fig = px.imshow(corr_matrix,
    #                 text_auto=True,  # Shows values on heatmap
    #                 aspect="auto",   # Adjust the aspect ratio
    #                 color_continuous_scale='Viridis',  # Color scheme
    #                 labels=dict(color="Correlation"))  # Label for color bar

    # # Display the heatmap in Streamlit
    #     st.plotly_chart(fig)
    if st.checkbox("View PPS heatmap"):
    # Path to your heatmap image
        heatmap_image_path = 'hp.png'  # Update this path with your actual image path
    
    # Display the image of the heatmap
        st.image(heatmap_image_path, caption='PPS Heatmap', use_column_width=True)



    st.write("---")

    st.write(
        f"#### **Analysis of Most Correlated Features**\n"
        f"Based on the correlation and PPS analyses, we can observe the following:\n\n"
        f"1. **Number of Miles (nsmiles)**: This feature shows a strong positive correlation with fare_lg. "
        f"This is expected as longer flights typically cost more.\n\n"
        f"2. **Number of Passengers (passengers)**: There's a moderate positive correlation between the number "
        f"of passengers and fare_lg. This could indicate that popular routes (with more passengers) tend to have higher fares.\n\n"
        f"3. **Large Market Share (large_ms)**: This feature shows a notable correlation with fare_lg. "
        f"It suggests that airlines with a larger market share on a route might charge higher fares.\n\n"
        f"4. **Low-Fare Market Share (lf_ms)**: There's a negative correlation between low-fare market share and fare_lg. "
        f"This implies that routes with more low-cost carrier competition tend to have lower fares.\n\n"
        f"5. **Year**: The year shows some correlation with fare_lg, which could indicate a general trend of "
        f"increasing fares over time or reflect economic factors affecting air travel pricing.\n"
    )

    feature_distribution = st.selectbox(
        "Select feature to view distribution:",
        engineered_df.columns.tolist()
    )

    def plot_distribution(df, col):
        fig, ax = plt.subplots(figsize=(14, 8))
        if df[col].dtype == 'object':
            sns.countplot(data=df, x=col, ax=ax)
        else:
            sns.histplot(data=df, x=col, ax=ax)
        plt.xticks(rotation=90)
        plt.title(f"Distribution of {col}", fontsize=20, y=1.05)
        st.pyplot(fig)

    plot_distribution(engineered_df, feature_distribution)

    st.write("---")

    st.write(
        f"#### **Feature Relationships**\n"
        f"* A parallel plot is used to visualize relationships between multiple features simultaneously.\n"
        f"* This plot helps us see patterns and trends across different variables.\n"
        f"* Pay attention to how the lines flow between different axes to identify potential relationships.\n"
    )


    import pandas as pd
    import plotly.express as px
    import streamlit as st

# Define a function to prepare the data
    def prepare_parallel_plot_data(df):
    # Take the first 1000 rows
        df = df.head(1000)
    
    # Apply pd.qcut on columns of type float64
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.qcut(df[col], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
        return df

# Load the cleaned dataset
    data_file = 'data_file.csv'
    cleaned_df = pd.read_csv(data_file)

# Prepare the data for parallel plot
    parallel_df = prepare_parallel_plot_data(cleaned_df)

# Create a color mapping for the fare column
    color_mapping = {
    'Low': '#1f77b4',        # Blue
    'Medium-Low': '#ff7f0e', # Orange
    'Medium-High': '#2ca02c',# Green
    'High': '#d62728'        # Red
}

# Apply color mapping to the 'fare' column
    parallel_df['fare_color'] = parallel_df['fare'].map(color_mapping)

# Create the parallel plot using plotly express
    fig = px.parallel_categories(
    parallel_df, 
    dimensions=parallel_df.columns.drop(['fare', 'fare_color']),  # Excluding 'fare' from dimensions
    color=parallel_df['fare_color'],  # Use the color-mapped 'fare_color' column
    color_continuous_scale=px.colors.sequential.Viridis
)

# Displaying the plot in Streamlit if the checkbox is selected
    if st.checkbox("View parallel plot"):
        st.plotly_chart(fig)



    st.write("---")

    st.success(
        f"#### **Conclusions**\n\n"
        f"Based on our correlation study, PPS analysis, and feature analysis, we can draw the following conclusions:\n\n"
        f"1. **Distance is a Key Factor**: The number of miles (nsmiles) shows the strongest correlation with fare, "
        f"confirming that distance is a primary driver of airline pricing.\n\n"
        f"2. **Market Dynamics Matter**: Both large market share (large_ms) and low-fare market share (lf_ms) "
        f"significantly influence fares. This suggests that competitive landscapes on different routes play a crucial role in pricing.\n\n"
        f"3. **Demand Influences Price**: The moderate correlation between number of passengers and fare indicates "
        f"that popular routes might command higher prices, possibly due to higher demand.\n\n"
        f"4. **Temporal Trends**: The correlation with the 'Year' variable suggests that there might be long-term "
        f"trends in fare pricing, possibly reflecting economic changes or industry developments.\n\n"
        f"5. **Complex Interactions**: The parallel plot and PPS analysis reveal complex interactions between variables. "
        f"These non-linear relationships provide additional insights beyond simple correlations.\n\n"
        f"These findings can significantly influence our fare prediction model:\n"
        f"- We should ensure that distance (nsmiles) is a key feature in our model.\n"
        f"- Market share variables should be included to capture competitive dynamics.\n"
        f"- Consider creating interaction terms, especially between distance and passenger numbers.\n"
        f"- Time-based features (like year and quarter) should be included to capture temporal trends.\n"
        f"- Feature engineering, such as creating a 'price per mile' feature, could provide additional insights.\n"
        f"- The PPS analysis highlights potential non-linear relationships that might benefit from more complex modeling techniques.\n\n"
        f"Further analysis could involve segmenting the data by different categories (e.g., short-haul vs long-haul flights) "
        f"to see if these relationships hold across different subsets of the data."
    )
