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



def train_validate_test_split(df, seed=123):
    train_and_validate, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df.survived
    )
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify=train_and_validate.survived,
    )
    return train, validate, test
