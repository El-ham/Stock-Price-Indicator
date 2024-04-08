# Importing Required Python Libraries
import yfinance as yf
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from datetime import datetime
import os


def Collect_Stock_Data(start_date, end_date, list_of_ticker_symbols):

    """
    Collects historical stock data for a given list of ticker symbols within a specified date range.

    This function iterates through each ticker symbol, retrieves its historical market data using
    the yfinance library, and stores the data in a dictionary if it meets certain conditions. It checks
    whether the data is not empty and if the number of rows meets at least 80% of the expected number of
    days within the given timeframe, assuming 250 trading days in a year.

    Parameters:
    - start_date (str): The start date for the data collection in "YYYY-MM-DD" format.
    - end_date (str): The end date for the data collection in "YYYY-MM-DD" format.
    - list_of_ticker_symbols (list): A list of string ticker symbols for which to collect data.

    Returns:
    - dict: A dictionary with ticker symbols as keys and their corresponding historical data DataFrames as values.

    Note:
    The function prints messages indicating the success of data collection for each ticker or the failure
    to meet the data requirements.
    """
    
    # Dictionary to store the historical data DataFrames
    dataframes= {}
    
    # Extracting Stock Data
    for stock in list_of_ticker_symbols:        

        stk = yf.Ticker(stock)
        
        current_data = stk.history(start=start_date, end=end_date)
        
        # Calculate the number of rows and check against the 80% threshold of expected days
        number_rows = current_data.shape[0]
        expected_days = (250/365)*((datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days)*0.8
        
        if (current_data.empty==False) and (number_rows>=expected_days):
            print(stock, 'data has been collected successfully.')
            # Store the DataFrame
            dataframes[stock] = current_data
        else:
            print('Sorry we can not do prediction for {}, Since No data found for this stock in this time frame.'.format(stock))
            
    return dataframes

def Drop_Continuous_NaNs(df, max_allowed_continuous_nans=3):
    
    """
    Filters out rows from a pandas DataFrame that are part of a continuous sequence of NaNs (Not a Number)
    exceeding a specified threshold in any column.

    The function identifies sequences of continuous NaNs across all columns and drops any rows that are
    part of a sequence longer than `max_allowed_continuous_nans`. It handles each column independently,
    ensuring that no column has continuous NaNs exceeding the threshold.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be processed.
    - max_allowed_continuous_nans (int, optional): The maximum allowed length of continuous NaN sequences.
      Rows that are part of a sequence longer than this will be dropped. Defaults to 3.

    Returns:
    - pandas.DataFrame: A DataFrame with rows in continuous NaN sequences beyond the allowed threshold removed.

    Note:
    The function directly analyzes the NaN presence in each row, groups continuous NaN sequences,
    and filters based on the specified threshold. It preserves the original DataFrame structure
    and data integrity, except for the removal of specified sequences.
    """

    # Identify rows that are completely NaN
    all_nan_rows = df.isna().all(axis=1)
    
    # Group consecutive NaN rows to identify continuous sequences
    all_nan_groups = (~all_nan_rows).cumsum()
    
    # Calculate the size of each group of continuous NaN rows
    group_sizes = all_nan_groups.map(all_nan_groups.value_counts())
    
    # Generate a mask for valid rows: either not part of a NaN sequence or within the allowed NaN sequence length
    valid_rows_mask = (group_sizes <= max_allowed_continuous_nans) | (~all_nan_rows)
    
    # Apply the mask to filter the DataFrame, removing unwanted continuous NaN sequences
    df_filtered = df[valid_rows_mask]
    
    return df_filtered

def Clean_NaN_Values(dataframes):
    
    """
    Cleans NaN values across a collection of pandas DataFrames by first removing rows with continuous
    NaNs beyond a set threshold, then addressing remaining NaNs in specific columns through either deletion,
    filling, or imputation based on the nature of the data and the proportion of missing values.

    Parameters:
    - dataframes (dict): A dictionary where keys are identifiers (e.g., ticker symbols) and values are
      pandas DataFrames containing stock market data.

    Returns:
    - dict: The same dictionary with DataFrames cleaned of NaN values according to the specified rules.

    Note:
    This function specifically treats financial data columns differently based on their importance and
    the feasibility of imputation. It also provides warnings for DataFrames where critical columns like
    'Close' have a high percentage of NaN values, potentially indicating inadequate data for reliable analysis.
    """
    
    for ticker in dataframes:
        df = dataframes[ticker]

        # Drop rows with continuous NaNs longer than the threshold
        df = Drop_Continuous_NaNs(df, max_allowed_continuous_nans=3)    

        # Calculate the percentage of NaNs in the 'Close' column
        closing_price_NaN = df['Close'].isna().mean() * 100
        NaN_threshold_for_close = 12

        # Check if 'Close' column NaN percentage exceeds threshold
        if closing_price_NaN > NaN_threshold_for_close:
            print('Sorry we can not predict {} since more than {} of the Close Price column is NaN'.format(ticker, NaN_threshold_for_close))
            
        else:
            for column in df.columns:
                # Fill NaNs with 0 for 'Dividends' and 'Stock Splits'
                if column in ['Dividends', 'Stock Splits']:
                    df[column].fillna(0, inplace=True)

                # Analyze and handle NaNs for financial columns differently
                if column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    # NaN analysis on the cleaned DataFrame
                    nan_percent = df[column].isna().mean() * 100

                    # If >50% NaNs, drop the column with a warning
                    if nan_percent > 50:
                        
                            # Attempt to acquire more complete data or use domain knowledge
                            print(f"Column: {column} has >50% NaNs. Dropping this column. Consider acquiring more complete data.")
                            del df[column]
                    # For NaN percentages between 0 and 50
                    elif 0 < nan_percent <= 50:
                        # Use forward fill and backward fill for low NaN percentage
                        if nan_percent < 5: 
                            # Forward fill for financial time series data
                            df[column]= df[column].fillna(method='ffill').fillna(method='bfill')
                            
                        else:
                            # Apply KNN imputation for higher NaN percentages to the entire DataFrame (considering all columns)                            
                            columns_for_imputation = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                            knn_imputer = KNNImputer(n_neighbors=5)
                            imputed_data = knn_imputer.fit_transform(columns_for_imputation)
                            imputed_df = pd.DataFrame(imputed_data, columns=columns_for_imputation.columns)
                            
                            # Merge the imputed 'Volume' column back into the original DataFrame (Step 5)
                            df[column] = imputed_df[column]
                    
                    else:
                        # No action needed for columns without NaN values
                        pass
        # Update the DataFrame in the dictionary
        dataframes[ticker] = df

    return dataframes

def Calculate_Adjusted_Price(dataframes):

    """
    Calculates the adjusted closing price for stocks within each DataFrame in a collection, taking into account
    stock splits and dividends. The adjusted closing price reflects the stock's value after accounting for
    corporate actions.

    Adjustments are made by modifying the 'Adjusted_Close' column based on the 'Stock Splits' and 'Dividends'
    columns. Stock splits adjust the price directly, while dividends adjust the price to reflect the payout's
    impact on the stock's value.

    Parameters:
    - dataframes (dict): A dictionary of pandas DataFrames keyed by ticker symbols 

    Returns:
    - dict: The same dictionary with each DataFrame's 'Adjusted_Close' column updated to reflect the adjusted
      closing prices.

    Note:
    The function processes each DataFrame independently, ensuring the date column is in the correct format
    and sorted in ascending order before making adjustments.
    """
    
    for ticker in dataframes:
        df = dataframes[ticker]
        
        # Ensure the 'Date' column is in 'YYYY-MM-DD' format and set as datetime type
        df.reset_index(inplace= True)
        df['Date'] = df['Date'].apply(lambda x: str(x).split(' ')[0])
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            
        df.sort_values('Date', ascending=True, inplace=True)
        
        # Initialize 'Adjusted_Close' column with existing close prices
        df['Adjusted_Close'] = df['Close']
        
        # Adjust 'Adjusted_Close' for stock splits
        for i in df[df['Stock Splits'] != 0].index:
            split_factor = df.at[i, 'Stock Splits']
            df.loc[:i-1, 'Adjusted_Close'] /= split_factor
        
        # Adjust 'Adjusted_Close' for dividends
        for i in df[df['Dividends'] != 0].index:
            dividend = df.at[i, 'Dividends']
            price = df.at[i-1, 'Close']
            Dividend_Multiplier = 1 - (dividend/price)
            df.loc[:i-1, 'Adjusted_Close'] *= Dividend_Multiplier
            
        dataframes[ticker] = df
        
    return dataframes

def Calculate_RSI(dataframes, column='Adjusted_Close', period=14):
    
    """
    Calculates the Relative Strength Index (RSI) for the specified column in each DataFrame within a
    collection. RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions in the price of a stock.

    The RSI is calculated using a specified period (typically 14 days) and is scaled from 0 to 100. An RSI
    above 70 is generally considered overbought, while an RSI below 30 is considered oversold.

    Parameters:
    - dataframes (dict): A dictionary of DataFrames keyed by ticker symbols
    - column (str, optional): The column name to calculate RSI for. Defaults to 'Adjusted_Close'.
    - period (int, optional): The period over which to calculate RSI, typically 14. Defaults to 14.

    Returns:
    - dict: The same dictionary of DataFrames, now including an 'RSI' column with the calculated RSI
      values for each stock.

    Note:
    This function iterates over each DataFrame, calculating the RSI based on price changes in the specified
    column. It handles positive and negative changes separately to compute average gains and losses, which
    are then used to derive the RSI values.
    """
    
    for ticker in dataframes:
        df = dataframes[ticker]

        # Calculate price difference from the previous day
        delta = df[column].diff(1)
        # Isolate gains and losses from the delta
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate the Relative Strength (RS)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        dataframes[ticker] = df

    return dataframes

def Calculate_MACD(dataframes, column='Adjusted_Close', short_period=12, long_period=26, signal_period=9):
    
    """
    Calculates the Moving Average Convergence Divergence (MACD) and its signal line for the specified
    column in each DataFrame within a collection. MACD is a trend-following momentum indicator
    that demonstrates the relationship between two moving averages of a stock's price. The MACD is
    calculated by subtracting the long-term Exponential Moving Average (EMA) from the short-term EMA,
    with the signal line being an EMA of the MACD itself.

    Parameters:
    - dataframes (dict): A dictionary of DataFrames keyed by ticker symbols).
    - column (str, optional): The column name to calculate MACD for. Defaults to 'Adjusted_Close'.
    - short_period (int, optional): The period for the short-term EMA. Defaults to 12.
    - long_period (int, optional): The period for the long-term EMA. Defaults to 26.
    - signal_period (int, optional): The period for the signal line EMA. Defaults to 9.

    Returns:
    - dict: The same dictionary of DataFrames, now including 'MACD' and 'Signal_Line' columns with
      the calculated values.

    Note:
    This function adds two columns to each DataFrame: 'MACD', which is the difference between the short-term
    and long-term EMAs of the specified column, and 'Signal_Line', which is the EMA of the 'MACD' column over
    the specified signal period. These indicators are commonly used to identify potential trend changes and
    trading opportunities.
    """

    for ticker in dataframes:
        df = dataframes[ticker]
        
        # Calculate the short-term and long-term EMAs
        short_ema = df[column].ewm(span=short_period, adjust=False).mean()
        long_ema = df[column].ewm(span=long_period, adjust=False).mean()
        
        # Compute the MACD by subtracting the long-term EMA from the short-term EMA
        df['MACD'] = short_ema - long_ema
        
        # Calculate the signal line, which is the EMA of the MACD
        df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

        dataframes[ticker] = df

    return dataframes

def Calculate_Bollinger_Bands(dataframes, column='Adjusted_Close', period=20):
    
    """
    Calculates Bollinger Bands for the specified column in each DataFrame within a collection.
    Bollinger Bands are a volatility indicator and consist of three lines: the middle band is an N-period
    moving average (MA), the upper band is K standard deviations above the MA, and the lower band is K
    standard deviations below the MA. This function sets N to the specified period (commonly 20) and K
    to 2, following the standard practice.

    Parameters:
    - dataframes (dict): A dictionary of DataFrames keyed by ticker symbols.
    - column (str, optional): The column name to calculate Bollinger Bands for. Defaults to 'Adjusted_Close'.
    - period (int, optional): The period over which the moving average and standard deviation are calculated.
      Defaults to 20.

    Returns:
    - dict: The same dictionary of DataFrames, now including 'Upper_Band' and 'Lower_Band' columns with
      the calculated Bollinger Bands.

    Note:
    This function enhances each DataFrame with two new columns: 'Upper_Band' and 'Lower_Band', representing
    the calculated upper and lower Bollinger Bands, respectively. These bands are used to assess whether
    prices are high or low on a relative basis and can indicate potential market volatility or price
    consolidations.
    """

    for ticker in dataframes:
        df = dataframes[ticker]
        
        # Calculate the moving average (MA) and standard deviation (STD) for the specified period
        MA = df[column].rolling(window=period).mean()
        STD = df[column].rolling(window=period).std()

        # Compute the upper and lower Bollinger Bands
        df['Upper_Band'] = MA + (STD * 2)
        df['Lower_Band'] = MA - (STD * 2)

        dataframes[ticker] = df

    return dataframes

def Engineer_Features(dataframes):

    """
    Enriches each DataFrame with additional features useful for financial analysis and machine learning models.
    Features include moving averages, On-Balance Volume (OBV), future price targets for various horizons,
    and time-based features like day of the week and month.

    Parameters:
    - dataframes (dict): A dictionary of DataFrames keyed by ticker symbols.

    Returns:
    - dict: The same dictionary of pandas DataFrames, now augmented with additional features including
      moving averages (MA10, MA20), OBV, future targets (Target_1, Target_2, Target_7, Target_14, Target_28),
      and time-based features (Day_of_Week, Month).

    Note:
    The function aims to facilitate the identification of trends, prediction of future prices, and
    capturing of temporal patterns in stock data.
    """
    
    for ticker in dataframes:
        df = dataframes[ticker]

        # Calculate 10-day and 20-day moving averages
        df['MA10'] = df['Adjusted_Close'].rolling(window=10).mean()
        df['MA20'] = df['Adjusted_Close'].rolling(window=20).mean()
        
        # Compute On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Adjusted_Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Define target variables for various prediction horizons
        df['Target_1'] = df['Adjusted_Close'].shift(-1)
        df['Target_2'] = df['Adjusted_Close'].shift(-2)
        df['Target_7'] = df['Adjusted_Close'].shift(-7)
        df['Target_14'] = df['Adjusted_Close'].shift(-14)
        df['Target_28'] = df['Adjusted_Close'].shift(-28)

        # Extract day of the week and month from the 'Date' column for time-based features
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        
        dataframes[ticker] = df

    return dataframes

def Use_Market_Data(dataframes, start_date, end_date, NaN_threshold_for_close=10):

    """
    Retrieves, cleans, and processes historical market data for a set of predefined stock indices
    over a specified date range. The function filters the data based on a threshold for the percentage
    of NaN values allowed in the 'Close' column and the expected data completeness. It calculates adjusted
    closing prices for each index, merges the data into a single DataFrame, and fills any missing values.

    Parameters:
    - dataframes (dict): A dictionary to store the adjusted price DataFrames, keyed by index symbols.
    - start_date (str): The start date for the data retrieval in "YYYY-MM-DD" format.
    - end_date (str): The end date for the data retrieval in "YYYY-MM-DD" format.
    - NaN_threshold_for_close (int, optional): The maximum percentage of NaN values allowed in the 'Close'
      column before an index is excluded. Defaults to 10%.

    Returns:
    - pd.DataFrame: A DataFrame containing the adjusted closing prices for each index, with dates aligned
      and missing values interpolated.

    Note:
    The function leverages the yfinance library to fetch the data, filters based on the specified criteria,
    and uses custom logic to adjust prices for stock splits and dividends.
    """

    # Predefined stock indices symbols
    stock_indices_dict = {'Dow Jones Industrial Average': 'DJIA',
                          'Nasdaq Composite': 'COMP',
                          'Russell 2000': 'RUT',
                          'MSCI World Index': 'MSCI',
                          'FTSE 100 (Financial Times Stock Exchange 100 Index)':'UKX',
                          'Nikkei 225': '^N225',
                          'Shanghai Composite Index': '000001.SS',
                          'DAX': 'DAX',
                          'SPY': 'SPY',
                          'VIX': 'VIX'}

    # Dictionary to hold the cleaned and processed index data
    df_indices = {}
    
    # Iterate over each index symbol to fetch and clean data
    for symbol in stock_indices_dict:
        index = stock_indices_dict[symbol]
        ind = yf.Ticker(index)
        # Fetch historical market data using yfinance
        current_data = ind.history(start=start_date, end=end_date)

        # Calculate the percentage of NaN values in 'Close' column
        closing_NaN = current_data['Close'].isna().mean() * 100
        # Calculate the actual number of rows fetched
        number_rows = current_data.shape[0]
        # Calculate the expected number of rows based on date range and trading days 
        expected_days = ((250/365)*(datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days)*0.8
        
        # Check if data meets completeness criteria and NaN threshold
        if (number_rows>= expected_days) and (closing_NaN < NaN_threshold_for_close):
            
            # Select relevant columns and fill NaN values
            current_data = current_data[['Dividends', 'Stock Splits', 'Close']]
            current_data['Dividends'].fillna(0, inplace=True)
            current_data['Stock Splits'].fillna(0, inplace=True)
            current_data['Close'] = current_data['Close'].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            df_indices[index] = current_data
        else:
            print('No data for ', index, '!')

    # DataFrame to aggregate adjusted market data
    Adjusted_Market = pd.DataFrame()
    if len(df_indices) !=0:
        # Adjust prices for stock splits and dividends
        df_indices = Calculate_Adjusted_Price(df_indices)

        # Merge adjusted prices from each index into a single DataFrame
        for ticker in df_indices:
            df = df_indices[ticker]
            df.rename(columns = {'Adjusted_Close': str('Adjusted_Close'+'_'+ticker)}, inplace=True)
            
            if Adjusted_Market.empty:
                Adjusted_Market = df[['Date', str('Adjusted_Close'+'_'+ticker)]]
            else:
                Adjusted_Market = pd.merge(Adjusted_Market, df[['Date', str('Adjusted_Close'+'_'+ticker)]], on='Date', how='outer', suffixes=('', ''))

    # Interpolate missing values to ensure completeness
    Adjusted_Market = Adjusted_Market.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')   
    
    return Adjusted_Market

def Merge_stock_with_Market(dataframes, Adjusted_Market):

    """
    Merges adjusted market index data with each individual stock DataFrame in a collection,
    aligning them by date. This operation enriches stock data with broader market context,
    facilitating analyses that require comparing stock performance against market indices.

    Parameters:
    - dataframes (dict): A dictionary of DataFrames keyed by ticker symbols.
    - Adjusted_Market (pd.DataFrame): A DataFrame containing adjusted market index data, with one column
      for dates and other columns for the adjusted closing prices of various market indices.

    Returns:
    - dict: The original dictionary of DataFrames, now with each stock DataFrame merged with the
      Adjusted_Market DataFrame on the 'Date' column.

    Note:
    The function ensures that the merge operation retains all rows from the stock DataFrames (using a
    'left' join).
    """

    # Iterate over each stock DataFrame to merge with market data
    for ticker in dataframes:
        df = dataframes[ticker]
        
        # Check if the Adjusted_Market DataFrame contains data
        if not Adjusted_Market.empty:
            # Merge the stock DataFrame with Adjusted_Market on 'Date', appending market data
            df = pd.merge(df, Adjusted_Market, on='Date', how='left', suffixes=('', ''))
            # Update the dictionary with the merged DataFrame
            dataframes[ticker] = df
        
    return dataframes

def Build_DataFrames(dataframes, start_date, end_date):
    
    """
    Saves each DataFrame in the provided dictionary to a CSV file named after the ticker symbol.

    Parameters:
    - dataframes (dict): A dictionary where each key is a ticker symbol (string) and each value is a pandas DataFrame.
      The function iterates through this dictionary, saving each DataFrame to a CSV file.

    Note:
    This function does not return a value. Instead, it writes files to the disk, saving each DataFrame in the 'dataframes'
    dictionary to a separate CSV file.
    """
    
    for ticker in dataframes:
        # Concatenate the ticker symbol with '.csv' to form the filename        
        name_df = str(ticker+'_'+start_date+'_'+end_date+'.csv')
        
        # Get the current working directory (where your notebook is located)
        current_directory = os.getcwd()
        
        # Name of the subfolder where you want to save the CSV file
        subfolder_name = 'Dfs'

        # Construct the full path to the subfolder
        subfolder_path = os.path.join(current_directory, subfolder_name)
        
        # Check if the subfolder exists, if not, create it
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Full path for the CSV file
        file_path = os.path.join(subfolder_path, name_df)

        # Save the DataFrame associated with the current ticker to a CSV file, excluding the index
        dataframes[ticker].to_csv(file_path, index=False)

