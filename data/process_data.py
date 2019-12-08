import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data on messages and categories and returns a merged dataframe

    Input: 
    - messages_filepath
    - categories_filepath

    Output:
    - df: dataframe with merged data from messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True, subset='id') #some ids have multiple category entries, keep the first only
    
    df = messages.merge(categories)
    return df

def clean_data(df):
    '''
    Clean dataframe by splitting the categories column in multiple 
    columns filled with 0s and 1s

    Input: 
    - df: contains data on messages and categories

    Output:
    - df: cleaned dataframe
    '''
    categories = df['categories'].str.split(';', expand=True)
    
    #column names
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda i: i[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        #sometimes there are 2s instead of 1s, replace 2 with 1
        categories[column] = categories[column].str.replace('2','1')
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    #drop original categories column
    df.drop('categories', axis=1, inplace=True)
    
    #concatenate with new categories split columns
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Save the cleaned data on messages and categories in a database

    Input: 
    - df: the dataframe you want to save
    - database_filename: destination where to save it 

    No Output
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
