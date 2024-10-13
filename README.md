# US Airline Fare Prediction System

## Overview

This project is a [US Airline Fare Prediction System](https://fare-prediction-1.onrender.com) built using Python, Streamlit, and various machine learning libraries. It allows users to predict airline ticket prices based on different flight routes and relevant features. The application is designed to help users understand fare trends and make predictions based on historical data of US airline flight routes and fares from 1993 to 2024.

## Features

- Predicts airline fares for different routes.
- Visualizes fare trends and patterns using interactive charts.
- Allows users to explore the relationship between various features such as distance, carrier, and fare prices.
- Includes machine learning models to forecast ticket prices based on user input.

## Business Requirements

The application addresses the following business requirements through machine learning and data analysis:

- **User Story 1**: As a user, I want to predict fares for different flight routes to help me make informed ticket purchase decisions.
- **User Story 2**: As a business analyst, I want to understand fare trends over time so that I can identify pricing patterns.
- **User Story 3**: As an airline operator, I want to analyze fare variability between different carriers to identify potential competitive advantages.

Each user story is mapped to tasks like model development, data visualization, and feature importance analysis, as described below.

## Installation

To run the project locally, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fare-prediction
   ```

2. Navigate to the project directory:
   ```bash
   cd fare-prediction
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application using Streamlit:
   ```bash
   streamlit run src/app.py
   ```

6. Access the application at [http://localhost:8501](http://localhost:8501) in your browser.

## File Structure

- **src/**: Contains the main application code (`app.py`).
- **data/**: Includes the dataset (`data_file.csv`, `US Airline Flight Routes and Fares 1993-2024.csv`).
- **models/**: Contains the pre-trained model (`model.pkl`).
- **notebooks/**: Jupyter notebooks for model training and exploration.
- **static/**: Static assets such as images.
- **requirements.txt**: Lists all the necessary Python dependencies.

## Dataset

The dataset used in this project contains historical US airline flight routes and fares from 1993 to 2024. The dataset includes columns for:

- Year and quarter
- Origin and destination airports
- Number of miles between the airports
- Number of passengers
- Fare prices (for large carriers and low-cost carriers)
- Carrier information and other related features

## Model

The application uses a machine learning model (Linear Regression) for fare prediction. The model was trained using historical fare data and saved using Joblib. The model file (`model.pkl`) is loaded when the application runs to make fare predictions.

### Model Evaluation

The model's performance was evaluated using the following metrics:

- **R² score**: Measures how well the model predicts fare prices. The R² score for this model is 0.85, indicating a good fit.
- **Actual vs Predicted**: A plot showing the relationship between the actual and predicted fare prices.
- **Error Analysis**: An analysis of the errors in fare prediction to understand where the model performs well or where improvements can be made.

## How to Use the App

1. Enter flight details such as origin and destination airports, miles, carrier, etc.
2. Click "Predict Fare" to get the predicted fare for the given flight route.
3. Explore the fare trend visualizations to understand historical fare patterns.

## Data Visualizations

The dashboard includes several visualizations to help answer the project’s business requirements:

1. **Fare Trend Over Time**: A line chart showing how fares have changed over the years for different routes and carriers.
2. **Fare Distribution by Carrier**: A bar chart that shows the variability of fares between different carriers.
3. **Correlation Heatmap**: Displays the correlation between features such as distance, carrier, and fare prices.
4. **Fare vs Distance Scatter Plot**: Visualizes the relationship between fare prices and flight distance.

Each plot helps users to explore different aspects of the data and supports the predictions made by the model.

## CRISP-DM Process

This project follows the CRISP-DM methodology:

1. **Business Understanding**: Predict airline fares based on historical data.
2. **Data Understanding**: The dataset includes historical flight data from 1993 to 2024.
3. **Data Preparation**: Data was cleaned, missing values were handled, and relevant features were selected for the model.
4. **Modeling**: A Linear Regression model was developed and evaluated.
5. **Evaluation**: The model's performance was assessed based on R² score and error analysis.
6. **Deployment**: The model was deployed via Streamlit as an interactive web application.

## Git & Version Control

The project uses Git and GitHub for version control. Commits are made at every significant stage of the project (data cleaning, model training, and deployment).

## Conclusion

The US Airline Fare Prediction System successfully predicts fares for different flight routes and provides valuable insights through visualizations. The model has a good R² score and performs well in predicting fares, addressing the business requirements set out in the user stories.
