#this function is to load data 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

data_path = "/data/real_estate.csv"
def load_and_preprocess_data(data_path):
    
    try:
        
        # Import the data from 'age_salary.csv'
        df = pd.read_csv(data_path)

        # Impute all missing values in all the features
        df['basement'].fillna(0, inplace=True)
        # change the basement type to integer
        df.basement=df.basement.astype(int)
        #remove outliers 
        # Print top 5 homes with largest lot_size
        df.lot_size.sort_values().tail()
        
            # print the record where lot_size = 1220551    
        df[df.lot_size == 1220551]
            # Drop the row with lot_size = 1220551
        df = df.drop(102)
       
        return df
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))

if __name__=="__main__":
    df = load_and_preprocess_data(data_path)   
    print(df.head())     