Stock Price Prediction Pipeline README
Overview
This Python-based project offers a robust pipeline for stock market analysis, focusing on predicting future stock prices. Leveraging the yfinance library for historical stock data, it encompasses detailed data preparation, sophisticated feature engineering, and the application of multiple machine learning models. The pipeline is designed to be modular, with distinct components for data handling and model training, accompanied by a user-friendly script for interaction.

Installation
Before starting, ensure Python 3.x is installed on your machine. This project requires several external libraries, which can be installed using the following command:

bash
pip install pandas numpy yfinance scikit-learn xgboost statsmodels pickle-mixin
Project Structure
Data Preparation (Data_Preparation.py)
Purpose: Automates the retrieval and initial processing of stock data, including cleaning and feature engineering.
Functions:
Collect_Stock_Data: Fetches historical data from Yahoo Finance.
Clean_NaN_Values: Cleans datasets by handling missing values appropriately.
Calculate_Adjusted_Price: Adjusts stock prices for dividends and splits.
Technical indicators calculation: Includes RSI, MACD, and Bollinger Bands.
Engineer_Features: Enhances data with engineered features for improved model performance.
Key Benefits: Streamlines the often cumbersome process of data preparation, ensuring datasets are clean, comprehensive, and ready for modeling.
Modeling (Modeling.py)
Purpose: Constructs and evaluates predictive models, selecting the best performer for future predictions.
Features:
Multiple regression models: Linear Regression, OLS, XGBoost, Random Forest.
Model_with_*: Functions for each model type, encapsulating the training and evaluation process.
Save_Best_Model: Compares models based on Mean Squared Error (MSE) and R-squared metrics, selecting the best.
Make_Pickle: Serializes the best model for later use, facilitating easy prediction on new data.
Key Benefits: Offers flexibility in model choice and ensures the selection of the most accurate model for the data at hand.
Main Script
Purpose: Provides an interactive interface for the user to either train new models or make predictions using pre-trained models.
Functionality:
action1: Guides users through training models on chosen stock tickers and date ranges.
action2: Allows users to select a pre-trained model and make predictions.
User Interaction: Designed to be intuitive, prompting users at each step to input their choices, ensuring a seamless experience even for those less familiar with stock market analysis or machine learning.
Usage
To Train Models:

Execute the main script: python main.py
Select option 1 when prompted.
Enter the stock ticker(s) and the desired date range as instructed.
The pipeline will automatically process the data, train multiple models, and save the best-performing model for each prediction interval.
To Make Predictions:

Run the main script: python main.py
Choose option 2 when prompted.
Follow the prompts to select from available pre-trained models.
The script will use the selected model to predict future stock prices based on the most recent data.
Contributing
We warmly welcome contributions, whether they be in the form of bug fixes, feature additions, or documentation improvements. To contribute:

Fork the repository.
Create a new branch for your changes.
Submit a pull request with a clear description of your modifications.
License
Distributed under the MIT License. See LICENSE for more information.

Acknowledgments
Data provided by Yahoo Finance through the yfinance library.
All the open-source projects and libraries that made this project possible.