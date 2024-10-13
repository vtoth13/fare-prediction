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

