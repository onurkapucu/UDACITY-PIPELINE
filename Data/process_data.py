import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Loads the messages and categories csv datasets from their corresponding addresses and merges them into a single pandas dataframe referred as df
    Outputs merged dataframe.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # merge datasets
    categories =  pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """ This function 
    1- Split categories into separate category columns
    2- Convert category values to just numbers 0 or 1.
    3- Replace categories column in df with new category columns.
    4- Remove Duplicates
    
    Input is the dataframe merged in the load_data function.
    Output is cleaned dataframe.
    """
    
    ##  1- Split categories into separate category columns.

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # rename the columns of `categories`
    categories.columns = category_colnames

    ## 2- Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    ## 3- Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    ## 4- Remove duplicates.
    df=df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """ Saves the input DataFrame df to the Database address. 
    Inputs are cleaned dataframe and address of where database file will be saved.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DF', engine, index=False)  

def main():
    """ This module
    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database 
    
    Required inputs are filepaths for messages dataset, categories dataset and sql database consecutively.
    Messages and Categories datasets should be in csv format.
    """
    
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