import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

def Read_Data(ticker, start_date, end_date):

    """
    Reads data from a CSV file into a pandas DataFrame.
    
    This function constructs a file path from a specified ticker symbol, start date, and end date,
    then reads the corresponding CSV file from the 'Dfs' folder within the current working directory.
    
    Parameters:
    - ticker (str): The ticker symbol for the data file.
    - start_date (str): The start date of the data range.
    - end_date (str): The end date of the data range.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the data from the CSV file.
    
    """
    
    # Name of the folder where your CSV file is located
    folder_name = 'Dfs'
    
    # Construct the file name using the provided parameters.
    file_name = ticker+'_'+start_date+'_'+end_date+'.csv'
    
    # Retrieve the current working directory to build the full file path.
    current_directory = os.getcwd()
    
    # Construct the full path to the CSV file
    file_path = os.path.join(current_directory, folder_name, file_name)
    
    # Read and return the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    return df

def Finalize_df(df, ticker, start_date, end_date, interval):

    """
    Cleans the DataFrame by dropping specific columns and rows with NaN values, and identifies the target column.
    
    This function iterates through the DataFrame's columns to remove any that contain the word 'Target' 
    but do not match the specified interval. It then identifies the remaining 'Target' column and 
    removes any rows in the DataFrame that contain NaN values, ensuring data integrity for further analysis.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to be cleaned and finalized.
    - interval (int): The interval value used to identify the relevant 'Target' column to retain.
    
    Returns:
    - tuple: A tuple containing:
        - The name of the target column (str).
        - The cleaned DataFrame (pd.DataFrame) with irrelevant target columns removed and no NaN values.
    """
    
    # Identify columns to drop: those including 'Target' in their name but not matching the specified interval.
    columns_to_drop = [col for col in df.columns if 'Target' in col and 'Target_'+str(interval) != col]
    
    # Drop identified columns from the DataFrame.
    dft = df.drop(columns=columns_to_drop)
    
    # Extract the name of the remaining 'Target' column.
    Target = [col for col in dft.columns if 'Target' in col][0]

    # Get the last row of the DataFrame
    last_row = dft.tail(1)
    del last_row[Target]
    del last_row['Date']
    
    try:
        # Save the last row to a CSV file for future predictions
        last_row.to_csv(f"Last_Rows/last_row_of_{ticker}_{start_date}_{end_date}.csv", index=False)

    except FileNotFoundError:
        # Handle the case where the 'Last_Rows' directory does not exist.
        print("Error: The specified directory 'Last_Rows' does not exist.")
    
    # Remove rows with any NaN values across all columns to ensure data completeness.
    df_cleaned = dft.dropna()

    return Target, df_cleaned

def Split_train_test(df, Target):

    """
    Splits the DataFrame into training and testing datasets for features and target.

    The function separates the DataFrame into features (X) and target (y) datasets by dropping 
    the specified target column and any non-feature columns (e.g., 'Date'). It then splits these 
    datasets into training and testing sets using a predefined test size and random state to ensure 
    reproducibility.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset to be split.
    - Target (str): The name of the target column in the DataFrame.
    
    Returns:
    - tuple: Contains the training and testing sets for both features (X_train, X_test) and 
      target (y_train, y_test).
    """
    
    # Prepare the features dataset by dropping the target column and other non-numeric/non-feature columns.
    X = df.drop([Target, 'Date'], axis=1)
    # Extract the target column as the target dataset.
    y = df[Target]
    
     # Split the data into training and testing sets, allocating 20% of the data to the testing set and
    # setting a random state for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def Model_with_LR(X_train, X_test, y_train, y_test):

    """
    Initializes, fits, and evaluates a Linear Regression model using the provided training and testing datasets.
    
    This function creates a Linear Regression model, fits it to the training data, and then uses the model
    to make predictions on the testing set. It evaluates the performance of the model by calculating the
    mean squared error (MSE) and the R-squared (R2) score between the actual and predicted values for the target.
    
    Parameters:
    - X_train (pd.DataFrame): The training dataset containing the features.
    - X_test (pd.DataFrame): The testing dataset containing the features.
    - y_train (pd.Series): The training dataset containing the target variable.
    - y_test (pd.Series): The testing dataset containing the target variable.
    
    Returns:
    - tuple: Contains the fitted Linear Regression model, the MSE, and the R2 score.
    """
    
    # Initialize the Linear Regression model.
    LR = LinearRegression()
    # Fit the model to the training data.
    LR.fit(X_train, y_train)
    
    # Use the fitted model to make predictions on the test dataset.
    y_pred = LR.predict(X_test)
    
    # Evaluate the model's performance by calculating the Mean Squared Error and R-squared score.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return LR, mse, r2

def Model_with_OLS(X_train, X_test, y_train, y_test):

    """
    Initializes, fits, and evaluates an Ordinary Least Squares (OLS) regression model.
    
    This function extends the X_train and X_test datasets with a constant term to account for the intercept 
    in the OLS model, fits the model to the training data, and evaluates its performance on the testing set 
    by calculating both the mean squared error (MSE) and R-squared (R2) score.
    
    Parameters:
    - X_train (pd.DataFrame): Training dataset containing the features.
    - X_test (pd.DataFrame): Testing dataset containing the features.
    - y_train (pd.Series): Training dataset containing the target variable.
    - y_test (pd.Series): Testing dataset containing the target variable.
    
    Returns:
    - tuple: Contains the fitted OLS model, the MSE, and the R2 score of the model on the testing set.
    """
    
    # Extend the features matrix of the training set with a column for the constant term.
    X_with_const = sm.add_constant(X_train)
    
    # Fit the OLS (Ordinary Least Squares) model
    ols = sm.OLS(y_train, X_with_const).fit()
    
    # Extend the features matrix of the testing set with a column for the constant term.
    X_test_with_const = sm.add_constant(X_test)
    # Use the fitted model to make predictions on the extended testing set.
    y_pred = ols.predict(X_test_with_const)
    
    # Evaluate the performance of the model on the testing set.
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return ols, mse, r2

def Model_with_XGB(X_train, X_test, y_train, y_test):

    """
    Initializes, fits, and evaluates an XGBoost regression model.
    
    This function creates an XGBoost regression model with specified hyperparameters, fits it to the 
    training data, and uses it to make predictions on the testing set. It evaluates the model's performance
    by calculating the mean squared error (MSE) and R-squared (R2) score between the actual and predicted
    values for the target variable.
    
    Parameters:
    - X_train (pd.DataFrame): The training dataset containing the features.
    - X_test (pd.DataFrame): The testing dataset containing the features.
    - y_train (pd.Series): The training dataset containing the target variable.
    - y_test (pd.Series): The testing dataset containing the target variable.
    
    Returns:
    - tuple: Contains the fitted XGBoost model, the MSE, and the R2 score.
    """
    
    # Initialize the XGBoost regressor model with specified hyperparameters.
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit the XGBoost model to the training data.
    xgb_model.fit(X_train, y_train)
    
    # Use the trained model to make predictions on the test dataset.
    xgb_pred = xgb_model.predict(X_test)
    
    # Evaluate the model's performance on the test data using Mean Squared Error and R-squared metrics.
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)
    
    return xgb_model, xgb_mse, xgb_r2

def Model_with_RF(X_train, X_test, y_train, y_test):

    """
    Initializes, fits, and evaluates a Random Forest regression model.
    
    This function creates a Random Forest regression model with default hyperparameters, fits it to the
    training data, and uses it to make predictions on the testing set. It evaluates the model's performance
    by calculating the mean squared error (MSE) and R-squared (R2) score between the actual and predicted
    values for the target variable.
    
    Parameters:
    - X_train (pd.DataFrame): The training dataset containing the features.
    - X_test (pd.DataFrame): The testing dataset containing the features.
    - y_train (pd.Series): The training dataset containing the target variable.
    - y_test (pd.Series): The testing dataset containing the target variable.
    
    Returns:
    - tuple: Contains the fitted Random Forest model, the MSE, and the R2 score.
    """
    
    # Initialize and fit the Random Forest regressor model to the training data.
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    
    # Use the trained model to make predictions on the test dataset.
    y_pred_rf = rf_model.predict(X_test)
    
    # Evaluate the model's performance on the test data using Mean Squared Error and R-squared metrics.
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    return rf_model, mse_rf, r2_rf

def Save_Best_Model(X_train, X_test, y_train, y_test):

    """
    Evaluates multiple regression models on the given dataset, identifies the best performing model based on
    mean squared error (MSE) and R-squared (R2) metrics, and pickles the best model if its R2 score is above 0.90.
    
    The function considers Linear Regression, OLS, XGBoost, and Random Forest models, comparing their performance
    on the given training and testing datasets. The best model is determined by the lowest MSE, with a tiebreaker
    based on the highest R2 score.
    
    Parameters:
    - X_train (pd.DataFrame): The training dataset containing the features.
    - X_test (pd.DataFrame): The testing dataset containing the features.
    - y_train (pd.Series): The training dataset containing the target variable.
    - y_test (pd.Series): The testing dataset containing the target variable.
    
    Returns:
    - The tuple of the best model, its MSE, and R2 score, if its R2 score is above 0.90; otherwise, None.
    """
    
    # Initialize variables to store the best model and its metrics
    best_model = None
    best_mse = float('inf')  # Initialize with a large value
    best_r2 = -float('inf')   # Initialize with a small value

    # Generate models using predefined functions.
    Model_LR = Model_with_LR(X_train, X_test, y_train, y_test)
    Model_OLS = Model_with_OLS(X_train, X_test, y_train, y_test)
    Model_XGB = Model_with_XGB(X_train, X_test, y_train, y_test)
    Model_RF = Model_with_RF(X_train, X_test, y_train, y_test)

    # List of models to evaluate
    models = {
        "Linear Regression": Model_LR,
        "OLS": Model_OLS,
        "XGBoost": Model_XGB,
        "Random Forest": Model_RF
    }

    # Evaluate each model, updating the best model based on MSE and R2.
    for model_name, (model, mse, r2) in models.items():

        # Check if the current model has better performance than the best model found so far
        if mse < best_mse:
            best_model = model_name
            best_mse = mse
            best_r2 = r2

    # Conditionally choose the best model if its performance is sufficiently high.
    if best_r2>=0.90:
        
        if best_model == "Linear Regression":
            model_to_pickle = Model_LR[0]
        elif best_model == "OLS":
            model_to_pickle = Model_OLS[0]
        elif best_model == "XGBoost":
            model_to_pickle = Model_XGB[0]
        elif best_model == "Random Forest":
            model_to_pickle = Model_RF[0]

    else:
        print('Could not create model with high coverage (R^2 >= 0.9) for the given data. Consider changing input parameters.')
        
    return model_to_pickle, best_mse, best_r2

def Make_Pickle(model_to_pickle, interval, ticker, start_date, end_date):

    """
    Pickles and saves the provided model to a file, named according to the specified ticker and interval.
    
    The function writes the model to a binary file within the 'Pickles' directory. The naming convention for the file
    includes the model's associated ticker symbol and interval, ensuring easy identification and retrieval of the model
    for future predictions or analysis.
    
    Parameters:
    - model_to_pickle: The model object to be pickled and saved.
    - interval (str or int): The interval associated with the model, contributing to the file's name.
    - ticker (str): The ticker symbol associated with the model, contributing to the file's name.
    
    Outputs:
    - On success, the model is saved to a .pkl file in the 'Pickles' directory, and a confirmation message is printed.
    - On failure, an error message is printed detailing the nature of the problem encountered.

    Raises:
    - FileNotFoundError: If the 'Pickles' directory does not exist.
    - IOError: If there is an issue writing to the file.
    - pickle.PicklingError: If the model object cannot be serialized.
    - Exception: Catches any other unexpected errors that occur during execution.
    
    """
    # Open the file and pickle the model object to the file.
    try:
        with open(f"Pickles/best_model_{ticker}_{interval}_{start_date}_{end_date}.pkl", "wb") as f:
            # Serialize and save the 'model_to_pickle' object into the file.
            pickle.dump(model_to_pickle, f)
        # Print a success message indicating the model has been successfully pickled and saved.    
        print(f"Model pickled successfully for {ticker}_{interval}.")
    except FileNotFoundError:
        # Handle the case where the 'Pickles' directory does not exist.
        print("Error: The specified directory 'Pickles' does not exist.")
    except IOError:
        # Handle general input/output errors, such as failure in file writing.
        print("IOError: There was an issue writing to the file.")
    except pickle.PicklingError:
        # Handle errors specific to the pickling process, such as if the object cannot be serialized.
        print("PicklingError: There was an issue pickling the model.")
    except Exception as e:
        # Catch any other unexpected exceptions and print an error message detailing the problem.
        print(f"An unexpected error occurred: {e}")

def Make_Predictions(ticker, start_date, end_date):

    """
    Processes stock data for a given ticker and date range to train, evaluate, and save the best predictive model
    for various time intervals.
    
    For each specified interval, this function reads historical stock data, cleans it, splits it into training and testing
    sets, identifies the best model based on performance metrics, and pickles this model for future use.
    
    Parameters:
    - ticker (str): The stock ticker symbol for which data is processed.
    - start_date (str): The start date for the data range to be processed.
    - end_date (str): The end date for the data range to be processed.
    
    Processes:
    - For each interval in a predefined list, performs data cleaning, model training and evaluation,
      and pickles the best-performing model.
    
    Outputs:
    - Saves pickled models for each interval to the disk.
    """

    # Define the list of intervals to generate models for.
    List_of_intervals = [1, 2, 7, 14, 28]

    # Read the data for the specified ticker and date range.
    df = Read_Data(ticker, start_date, end_date)
    print(ticker,'\'s Data frame has been built Successfully!')

    # Iterate over each interval, process the data, and save the best model.
    for interval in List_of_intervals: 
        # Prepare the data for the given interval.
        Target, df_cleaned = Finalize_df(df, ticker, start_date, end_date, interval)
        # Split the data into training and testing sets.
        X_train, X_test, y_train, y_test = Split_train_test(df_cleaned, Target)
        # Identify and save the best model for the interval.
        model_to_pickle, best_mse, best_r2 = Save_Best_Model(X_train, X_test, y_train, y_test)
        # Check if a model was selected to be pickled before attempting to pickle it.
        if model_to_pickle:
            print(f'Best model selected for {ticker} for predicting {interval} days ahead, with MSE = {best_mse} and Rsquared = {best_r2}')
            Make_Pickle(model_to_pickle, interval, ticker, start_date, end_date)
        else:
            print(f"No suitable model found for interval {interval} to pickle.")

