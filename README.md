# Stock Price Prediction Pipeline README

## Overview
This Python-based project provides a comprehensive pipeline for stock market analysis, focusing on predicting future stock prices. Utilizing the `yfinance` library for historical stock data, the pipeline includes extensive data preparation, advanced feature engineering, and the deployment of various machine learning models. It's designed with modularity in mind, featuring separate components for data management and model training, alongside an intuitive script for user interaction. For additional details and deeper insights about this project, please explore the Jupyter notebooks.

## Installation
Ensure Python 3.x is installed on your system. The project depends on multiple external libraries, installable via:

bash
pip install pandas numpy yfinance scikit-learn xgboost statsmodels pickle-mixin


## Project Structure

### Data Preparation (`Data_Preparation.py`)
**Purpose:** Simplifies the retrieval and initial processing of stock data, including data cleaning and feature engineering.

**Functions:**
- `Collect_Stock_Data`: Fetches historical data from Yahoo Finance.
- `Clean_NaN_Values`: Cleans data by addressing missing values.
- `Calculate_Adjusted_Price`: Adjusts stock prices for dividends and splits.
- `Technical Indicators Calculation`: Computes RSI, MACD, and Bollinger Bands.
- `Engineer_Features`: Adds engineered features to enhance model performance.

**Key Benefits:** Makes data preparation less tedious, ensuring datasets are clean, comprehensive, and modeling-ready.

### Modeling (`Modeling.py`)
**Purpose:** Builds and assesses predictive models, selecting the most accurate for future predictions.

**Features:**
- Multiple regression models: Linear Regression, OLS, XGBoost, Random Forest.
- `Model_with_*`: Specific functions for each model, covering training and evaluation.
- `Save_Best_Model`: Determines the best model based on MSE and R-squared metrics.
- `Make_Pickle`: Serializes the top model for future use, allowing predictions on new data.

**Key Benefits:** Facilitates flexible model selection, ensuring optimal accuracy.

### Main Script
**Purpose:** Offers an interactive interface for training models or making predictions with pre-trained ones.

**Functionality:**
- `action1`: Assists users in training models with selected stock tickers and date ranges.
- `action2`: Enables predictions using available pre-trained models.

**User Interaction:** Designed for ease of use, with prompts guiding users through each step.

## Usage

### To Train Models:
1. Execute the main script: `python Main.py`
2. Choose option 1 as prompted.
3. Input stock ticker(s) and date range as directed. The pipeline processes the data, trains models, and saves the best for each prediction interval.

### To Make Predictions:
1. Run the main script: `python Main.py`
2. Select option 2 when prompted.
3. Follow instructions to choose a pre-trained model for future stock price predictions.

## Contributing
Contributions are encouraged, whether bug fixes, features, or documentation improvements.

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed description of your changes.

## Blog Post
Here is the link to the blog post related to this analysis: [Predicting the Future: A Journey through Stock Price Forecasting with Python](https://medium.com/@e.nabipoor/predicting-the-future-a-journey-through-stock-price-forecasting-with-python-f97b95fcfb36)

## Acknowledgments
- My heartfelt gratitude to the Udacity organization for providing me with the opportunity to tackle this fascinating problem.
- Data sourced from Yahoo Finance via the `yfinance` library.
- Gratitude to all open-source projects and libraries enabling this work.
