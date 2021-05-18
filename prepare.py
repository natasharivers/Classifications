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

