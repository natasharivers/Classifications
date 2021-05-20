def prep_iris(df):
    '''
    accepts the untransformed iris data
    returns: the data with cleaning operations performed on it 
    '''
    df = df.drop(columns= (['species_id', 'measurement_id']))
    df = df.rename(columns = {'species_name': 'species'})
    dummy_name = pd.get_dummies(df[['species']], dummy_na = False, drop_first=[True])
    df = pd.concat([df, dummy_name], axis=1)
    return df



def titanic_split(df):
    '''
    This function performs split on titanic data, stratify survived.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.survived)
    
    return train, validate, test




def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df.survived)
    train, validate = train_test_split(train_and_validate,test_size=0.3, random_state=seed, stratify=train_and_validate.survived)
    
    return train, validate, test


def prep_titanic (df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    Returns prepped train, validate, and test dfs with embarked dummy vars,
    deck dropped, and the mean of age imputed for Null values.
    '''
    
    # drop rows where embarked/embark town are null values
    df = df[~df.embarked.isnull()]
    
    # encode embarked using dummy columns
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    
    # join dummy columns back to df
    df = pd.concat([df, titanic_dummies], axis=1)
    
    # drop the deck column
    df = df.drop(columns='deck')
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)
    
    # impute mean of age into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test
