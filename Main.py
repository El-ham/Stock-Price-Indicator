import sys
import os
from datetime import datetime
import re
import pandas as pd
import pickle



from Data_Preparation import Collect_Stock_Data, Drop_Continuous_NaNs, Clean_NaN_Values, Calculate_Adjusted_Price, Calculate_RSI, Calculate_MACD, Calculate_Bollinger_Bands, Engineer_Features, Use_Market_Data, Merge_stock_with_Market, Build_DataFrames

from Modeling import Read_Data, Finalize_df, Split_train_test, Model_with_LR, Model_with_XGB, Model_with_RF, Save_Best_Model, Make_Pickle, Make_Predictions

def action1():

    list_of_ticker_symbols =[]
    flag=1
    while flag != 0:
        ticker= input('Please enter tickers you want to train model one by one as many as you want. please enter 0 when you are done!')
        if ticker != str(0):
            list_of_ticker_symbols.append(str(ticker))
        else:
            flag= 0



    def is_valid_date_format(date_str):
        # Define a regular expression pattern for 'YYYY-MM-DD' format
        pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        return bool(pattern.match(date_str))

    while True:
        # Prompt the user to enter a date
        start_date = input("Enter a start date in the format 'YYYY-MM-DD' (Enter '0' to stop): ")
    
        if start_date == '0':
            print("Exiting the program.")
            return 0  # Exit the loop if the user enters '0'
    
        if is_valid_date_format(start_date):
            print("Valid date format:", start_date)
            break  # Exit the loop if a valid date is entered
        else:
            print("Invalid date format. Please enter a start date in the format 'YYYY-MM-DD'.")

    
    while True:
        # Prompt the user to enter a date
        end_date = input("Enter an end date in the format 'YYYY-MM-DD' (Enter '0' to stop): ")
    
        if end_date == '0':
            print("Exiting the program.")
            return 0  # Exit the loop if the user enters '0'
    
        if is_valid_date_format(end_date):
            print("Valid date format:", end_date)
            break  # Exit the loop if a valid date is entered
        else:
            print("Invalid date format. Please enter an end date in the format 'YYYY-MM-DD'.")


    dataframes = Collect_Stock_Data(start_date, end_date, list_of_ticker_symbols)
    list_of_ticker_symbols = list(dataframes.keys())

    dataframes = Clean_NaN_Values(dataframes)
    if dataframes is None:
        print('Error cleaning the data! Exiting now!')
        sys.exit(1)  # Exit with a non-zero exit code to indicate failure
    
    else:
        print('Data Cleaned Successfully!')

    
    dataframes = Calculate_Adjusted_Price(dataframes)
    print('Adjusted Price has been calculated Successfully!')
    
    dataframes = Calculate_RSI(dataframes, column='Adjusted_Close', period=14)
    print('RSI has been calculated Successfully!')
    
    dataframes = Calculate_MACD(dataframes, column='Adjusted_Close', short_period=12, long_period=26, signal_period=9)
    print('MACD has been added Successfully!')
    
    dataframes = Calculate_Bollinger_Bands(dataframes, column='Adjusted_Close', period=20)
    print('Bollinger_Bands has been calculated Successfully!')
    
    dataframes = Engineer_Features(dataframes)
    print('Feature engineering has been completed Successfully!')
    
    Adjusted_Market = Use_Market_Data(dataframes, start_date, end_date, NaN_threshold_for_close=10)
    print('Market data has been collected Successfully!')
    
    dataframes = Merge_stock_with_Market(dataframes, Adjusted_Market)
    print('Market data has been collected Successfully!')
    
    Build_DataFrames(dataframes, start_date, end_date)
    print('Market data has been collected Successfully!')


    for ticker in list_of_ticker_symbols:
        
        Make_Predictions(ticker, start_date, end_date)


def action2():

    directory="Pickles"
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        
    
    # List all pickle files in the directory that start with 'best_model_'
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl') and f.startswith('best_model_')]
    
    # Extract information from filenames
    files_info = []
    for filename in pickle_files:
        # Removing 'best_model_' prefix and '.pkl' suffix to extract the relevant parts
        parts = filename.replace('best_model_', '').replace('.pkl', '').split('_')
        if len(parts) == 4:
            #ToDo: correct the order of these parts
            ticker, interval, start_date, end_date = parts
            files_info.append([ticker, interval, start_date, end_date, filename])
    
    # Check if we found any files
    if not files_info:
        print("No pickle files found in the directory.")
    
    
    # Displaying the files in a tabular format
    print(f"{'Index':<5}{'Ticker':<10}{'Interval':<10}{'Start Date':<15}{'End Date':<15}")
    for index, (ticker, interval, start_date, end_date, _) in enumerate(files_info):
        print(f"{index:<5}{ticker:<10}{interval:<10}{start_date:<15}{end_date:<15}")
    
    # Prompt the user to select a file
    s = 0
    while s == 0:
        try:
            selected_index = int(input("Enter the index of the file you'd like to use. Please choose a value not in the list to exit: "))
            if 0 <= selected_index < len(files_info):
                selected_file = files_info[selected_index][-1]
                print(f"You selected: {selected_file}")
                pickle_for_prediction = os.path.join(directory, selected_file)
                s = 1
                # Read the last row 
                ticker = files_info[selected_index][0]
                interval = files_info[selected_index][1]
                start_date = files_info[selected_index][2]
                end_date = files_info[selected_index][3]
                
                last_row = pd.read_csv(f"Last_Rows/last_row_of_{ticker}_{start_date}_{end_date}.csv")
                # Read the pickle file
                with open(pickle_for_prediction, 'rb') as file:
                    loaded_model = pickle.load(file)
                # Make prediction with pickle file for the last row
                prediction = loaded_model.predict(last_row)
                # Show the result to the user
                print(f'The predicted price for {ticker} for the {interval} days after {end_date} equals to: {prediction}')
            else:
                print("Invalid index, exiting now.")
                sys.exit(0)  # Exit with a non-zero exit code to indicate failure
        except ValueError:
            print("Please enter a valid integer index.")
  

if __name__ == "__main__":

    action = 5
    while action !=3 :
        action = int(input('Please enter number 1, 2 or 3 to show which option you want to do:\n 1: train a model \n 2: make prediction \n 3: exit \n'))


        if action ==1 :
            action1()
        elif action ==2 :
            action2()
        elif action ==3 :
            sys.exit(0)

