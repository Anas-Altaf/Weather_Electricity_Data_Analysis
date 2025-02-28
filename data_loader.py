# data_loader.py
import glob
import json
import os

import pandas as pd
from tabulate import tabulate

raw_folder = os.path.join(os.getcwd(), 'raw', )


def get_all_files():
    elec_data_files = []
    weather_data_files = []
    all_data_folders = glob.glob(raw_folder + '/*')
    for folder in all_data_folders:
        if 'electricity' in folder:
            elec_data_files = glob.glob(folder + '/*')
        elif 'weather' in folder:
            weather_data_files = glob.glob(folder + '/*')
    return elec_data_files, weather_data_files


# print(f"Total number of files for Electricity Data: {len(elec_data_files)}")
# print(f"Total number of files for Weather Data: {len(weather_data_files)}")
#  Lets Work on Electricity Data
def get_elect_df():
    data_files = get_all_files()[0]
    elec_dfs = []
    df = pd.DataFrame()
    for file in data_files:
        # print(file)
        if file.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.endswith('.json'):
            json_data = json.load(open(file, encoding='utf-8'))
            df = pd.json_normalize(json_data['response']['data'], sep='_')
        elec_dfs.append(df)
    print(f'Total number of dataframes for Electricity Data: {len(elec_dfs)}')
    # Concatenate all the dataframes
    return pd.concat(elec_dfs, ignore_index=True)


# Handle Weather Data
def get_weather_df():
    data_files = get_all_files()[1]
    weather_dfs = []
    df = pd.DataFrame()
    for file in data_files:
        # print(file)
        if file.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.endswith('.json'):
            json_data = json.load(open(file, encoding='utf-8'))
            df = pd.json_normalize(json_data['response']['data'], sep='_')
        weather_dfs.append(df)
    print(f'Total number of dataframes for Weather Data: {len(weather_dfs)}')
    # Concatenate all the dataframes
    return pd.concat(weather_dfs, ignore_index=True)



def log_data(elect_data, weather_data):

    # Logging the Records
    total_records_elect = elect_data.shape[0]
    total_records_weather = weather_data.shape[0]
    num_columns_elect = elect_data.shape[1]
    num_columns_weather = weather_data.shape[1]
    columns_elect = ','.join(elect_data.columns.to_list())
    columns_weather = ','.join(weather_data.columns.to_list())
    headers = ['Name','Total Records', 'Total Columns', 'Columns', 'Data Types', 'Missing Values']
    # Data Types
    data_types_elect = elect_data.dtypes
    data_types_weather = weather_data.dtypes
    missing_values_elect = elect_data.isnull().sum()
    missing_values_weather = weather_data.isnull().sum()
    data = [
        ['Electricity',total_records_elect, num_columns_elect, columns_elect, data_types_elect, missing_values_elect],
        ['Weather',total_records_weather, num_columns_weather, columns_weather, data_types_weather, missing_values_weather],
    ]
    # using tabulate to print the data
    print(tabulate(data, headers=headers, tablefmt='fancy_grid'))

#  Logging
elect_data = get_elect_df()
weather_data = get_weather_df()
log_data(elect_data, weather_data)
input('Press Enter to Exit')