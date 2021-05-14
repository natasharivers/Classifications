import pandas
import numpy
import os
from env import host, user, password
from pydataset import data

################################GET CONNECTION HELPER FUNCTION###############################################


#helper function to get connection
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#######################################NEW TITANIC DATA HELPER FUNCTION#######################################################

 #helper function to get titanic_db
def new_titanic_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = 'SELECT * FROM passengers'
    return pd.read_sql(sql_query, get_connection('titanic_db')) 


##################################GET TITANIC DATA HELPER FUNCTION#############################################

def get_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in titanic df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('titanic_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    return df  


#######################################NEW IRIS DATA HELPER FUNCTION#######################################################
def new_iris_data():
    '''
    This function reads in the iris data from the Codeup db
    and returns a pandas DataFrame with all columns from 
    measurements and species tables.
    '''
    sql_query = 'SELECT * FROM measurements JOIN species ON measurements.species_id= species.species_id'
    return pd.read_sql(sql_query, get_connection('iris_db')) 


#######################################GET IRIS DATA HELPER FUNCTION#######################################################


def get_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in iris df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('iris_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_iris_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('iris_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    return df  