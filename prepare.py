import pandas as pd
import numpy as np

def prep_iris(df):
    """
    This function takes the iris data set (from aquire.py) as input.
    And outputs a clean dataset ready for modeling.
    Clean in this case means: species and measurement id columns removed, species_name changed to species and seperated 
    into dummy variables
    """
    cleaned_df = df.drop(columns = ['species_id', 'measurement_id'])
    cleaned_df = cleaned_df.rename(columns= {'species_name': 'species'})
    cleaned_df = pd.get_dummies(data=cleaned_df, columns= ['species'], drop_first = True)
    return cleaned_df


def impute_mode(df):
    '''
    impute mode for embark_town. Replaces missing values with most frequently occurring value.
    '''
    imputer = SimpleImputer(strategy='most_frequent', missing_values=None)
    df[['embark_town']] = imputer.fit_transform(df[['embark_town']])
    return df


def prep_titanic(df):
    '''
    takes in a dataframe of the titanic dataset as it is acquired and returns a cleaned dataframe
    argument: df: a pandas df with expected feature names and columns
    return: train, test, split: three dataframes with the cleanining operations performed on them
    '''
    #drop duplicates
    df = df.drop_duplicates()
    # drop cols we dont need in our model
    df = df.drop(columns= ['deck', 'embarked', 'class', 'age', 'passenger_id'])
    # replace missing values using imputer
    df = impute_mode(df)
    # divide df into a training and test dataset
    train, test = train_test_split(df, test_size= 0.2, random_state= 1349, stratify = df.survived)
    # divide train dataset into train and validate
    train, validate = train_test_split(train, train_size= 0.7, random_state=1349, stratify = train.survived)
    # change object dtype cols to integers
    dummy_train = pd.get_dummies(train[['sex','embark_town']], drop_first=[True, True])
    dummy_validate = pd.get_dummies(validate[['sex','embark_town']], drop_first=[True, True])
    dummy_test = pd.get_dummies(test[['sex','embark_town']], drop_first=[True, True])
    # merge each dummy df with appropriate dataset
    train = pd.concat([train, dummy_train], axis = 1)
    validate = pd.concat([validate, dummy_validate], axis = 1)
    test = pd.concat([test, dummy_test], axis = 1)
    # drop columns that have been converted to integers via dummy dfs
    train = train.drop(columns = ['sex','embark_town'])
    validate = validate.drop(columns = ['sex','embark_town'])
    test = test.drop(columns = ['sex','embark_town'])
    # return the three datasets
    return train, validate, test
