import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from CSV files and merge them into a single DataFrame.
    
    Args:
        messages_filepath (str): Filepath of the messages CSV file.
        categories_filepath (str): Filepath of the categories CSV file.
    
    Returns:
        df (pandas.DataFrame): Merged DataFrame.
    """
    # Load messages data
    messages = pd.read_csv(messages_filepath)
    
    # Load categories data
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    Clean the data by splitting the categories column, renaming columns, converting category values to numbers,
    dropping the original categories column, and dropping duplicates.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
    
    Returns:
        df_cleaned (pandas.DataFrame): Cleaned DataFrame.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Use this row to extract a list of new column names for categories
    category_colnames = row
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.head()
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filepath, table_name):
    """
    Save a DataFrame to an SQLite database using SQLAlchemy.
    
    Args:
        df (pandas.DataFrame): DataFrame to be saved.
        database_filepath (str): Filepath of the SQLite database.
        table_name (str): Name of the table to be created in the database.
        Disaster_Response (str): Name of the column in the table for disaster response data.
    """
    # Create an SQLAlchemy engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Save the DataFrame to the database
    df.to_sql(table_name, engine, index=False, if_exists='replace')





def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        table_name = 'disaster_response'  # Define the table name

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name)
        
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