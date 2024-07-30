import pandas as pd
import logging

# 
def create_features(df):
    try:
        # Create indicator variable for properties with 2 beds and 2 baths
        df['popular']= ((df.beds == 2)&(df.baths == 2)).astype(int)
        # Create a new variable recession
        df['recession'] = ((df.year_sold >= 2010) & (df.year_sold<=2013)).astype(int)
       # Create a property age feature
        df['property_age'] = df.year_sold - df.year_built
       # Remove rows where property_age is less than 0
        df = df[df.property_age >= 0]
       # Create dummy variables for 'property_type'
        df = pd.get_dummies(df, columns=['property_type'], drop_first=True)
        x=df.drop('price', axis=1)
        y=df['price']
        return x,y
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))