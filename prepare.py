from pydataset import data
import seaborn as sns
import pandas as pd
import numpy as np
import os
from env import host, user, password, get_db_url

def prep_iris(df):
    '''
    takes in a dataframe of the iris dataset as it is acquired and returns a cleaned dataframe
    arguments: df: a pandas DataFrame with the expected feature names and columns
    return: clean_df: a dataframe with the cleaning operations performed on it
    '''
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name': 'species'})
    dummy_df = pd.get_dummies(df[['species']], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df