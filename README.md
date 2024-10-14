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

## Technologies Used

The technologies used throughout the development are listed below:

### Languages

- [Python](https://www.python.org/)

### Python Packages

- [Pandas](https://pandas.pydata.org/) - Open source library for data manipulation and analysis.
- [NumPy](https://numpy.org/) - Adds support for large, multi-dimensional arrays and matrices, and high-level mathematical functions.
- [Matplotlib](https://matplotlib.org/) - Comprehensive library for creating static, animated and interactive visualisations.
- [Seaborn](https://seaborn.pydata.org/) - Data visualisation library for drawing attractive and informative statistical graphics.
- [Feature-engine](https://feature-engine.readthedocs.io/) - Library with multiple transformers to engineer and select features for machine learning models.
- [ppscore](https://github.com/8080labs/ppscore) - Library for detecting linear or non-linear relationships between two features.
- [scikit-learn](https://scikit-learn.org/) - Open source machine learning library that features various algorithms for training ML models.
- [Joblib](https://joblib.readthedocs.io/en/latest/) - Provides tools for lightweight pipelining, e.g., caching output values.
- [Plotly](https://plotly.com/python/) - A library for creating interactive visualizations in Python.

### Other Technologies

- [Git](https://git-scm.com/) - For version control.
- [GitHub](https://github.com/) - Code repository and project management through GitHub projects.
- [Render](https://www.render.com/) - For application deployment.
- [VSCode](https://code.visualstudio.com/) - IDE used for development.

[Back to top](#overview)

## Testing

### Manual Testing

#### User Story Testing

The dashboard was manually tested using user stories as a basis for determining success.
Jupyter notebooks were reliant on consecutive functions being successful, so manual testing against user stories was deemed less relevant.

**As a non-technical user, I can view a project summary that describes the project, dataset, and business requirements to understand the project at a glance.**

| Feature              | Action                  | Expected Result                                         | Actual Result          |
|----------------------|-------------------------|---------------------------------------------------------|------------------------|
| Project summary page  | Viewing summary page     | Page is displayed, can move between sections on page     | Functions as intended   |

**As a non-technical user, I can view the project hypotheses and validations to determine what the project was trying to achieve and whether it was successful.**

| Feature              | Action                  | Expected Result                                         | Actual Result          |
|----------------------|-------------------------|---------------------------------------------------------|------------------------|
| Project hypotheses page | Navigate to page       | Clicking on navbar link in sidebar navigates to correct page | Functions as intended   |

**As a non-technical user, I can enter unseen data into the model and receive a prediction (Business Requirement 2).**

| Feature              | Action                  | Expected Result                                         | Actual Result          |
|----------------------|-------------------------|---------------------------------------------------------|------------------------|
| Prediction page       | Navigate to page        | Clicking on navbar link in sidebar navigates to correct page | Functions as intended   |
| Enter live data       | Interact with widgets   | All widgets are interactive, respond to user input       | Functions as intended   |
| Live prediction       | Click 'Run Predictive Analysis' button | Clicking on button displays message on page with prediction and % chance | Functions as intended   |

**As a technical user, I can view the correlation analysis to see how the outcomes were reached (Business Requirement 1).**

| Feature              | Action                  | Expected Result                                         | Actual Result          |
|----------------------|-------------------------|---------------------------------------------------------|------------------------|
| Correlation Study page | Navigate to page       | Clicking on navbar link in sidebar navigates to correct page | Functions as intended   |
| Correlation data      | Tick correlation results checkbox | Correlation data is displayed on dashboard              | Functions as intended   |
| PPS Heatmap           | Tick PPS heatmap checkbox | Heatmap is displayed on dashboard                       | Functions as intended   |
| Feature Correlation   | Select feature from dropdown box | Relevant countplot is displayed                         | Functions as intended   |
| Parallel Plot         | Tick parallel plot checkbox | Parallel plot is displayed on dashboard, is interactive | Functions as intended   |

**As a technical user, I can view all the data to understand the model performance and see statistics related to the model (Business Requirement 2).**

| Feature              | Action                  | Expected Result                                         | Actual Result          |
|----------------------|-------------------------|---------------------------------------------------------|------------------------|
| Model performance page | Navigate to page       | Clicking on navbar link in sidebar navigates to correct page | Functions as intended   |
| Success metrics       | View page               | Success metrics outlined in business case are displayed  | Functions as intended   |
| ML Pipelines          | View page               | Both ML Pipelines from Jupyter notebooks are displayed   | Functions as intended   |
| Feature Importance    | View page               | Most important features are plotted and displayed        | Functions as intended   |
| Model Performance     | View page               | Confusion matrix for train and test sets are displayed   | Functions as intended   |

### Validation

All code in the `src` directory was validated as conforming to PEP8 standards.

Some files had warnings due to 'line too long', however, these were related to long strings when writing to the dashboard. These warnings were ignored as they did not affect the readability of any functions.

### Automated Unit Tests

No automated unit tests have been carried out at this time.

## Dashboard Design

The dashboard consists of the following pages, designed to answer the project’s business requirements and provide insights into fare prediction:

### Project Summary:
- Content: Introduction to the project, dataset details, and business requirements.
- Purpose: Helps non-technical users understand the project scope and its objectives.

### Fare Prediction:
- Content: Input fields for entering flight details such as origin, destination, miles, carrier, etc.
- Purpose: Allows users to enter unseen data and receive fare predictions.
- Business Requirement: Answers the need for fare prediction functionality.

### Fare Trends:
- Content: Visualizations like line charts and bar charts showing fare trends over time and between carriers.
- Purpose: Provides insights into how fares change over time or by carrier.
- Business Requirement: Addresses the requirement to explore fare variability.

### Correlation Study:
- Content: Correlation analysis, PPS heatmap, and parallel plots.
- Purpose: Helps technical users understand the relationships between features such as distance, carrier, and fare prices.
- Business Requirement: Provides correlation insights to meet analytical requirements.

### Model Performance:
- Content: Displays the model's performance metrics, including R² score, confusion matrix, and feature importance.
- Purpose: Allows technical users to evaluate model success.
- Business Requirement: Helps meet the business goal of accurate fare prediction.

Each page is navigable through the sidebar, and the design ensures that both non-technical and technical users can interact with the dashboard easily.
