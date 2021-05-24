def model_performs (X_df, y_df, model):
    '''
    Take in a X_df, y_df and model  and fit the model , make a prediction, calculate score (accuracy), 
    confusion matrix, rates, clasification report.
    X_df: train, validate or  test. Select one
    y_df: it has to be the same as X_df.
    model: name of your model that you prevously created 
    
    Example:
    mmodel_performs (X_train, y_train, model1)
    '''
    # create the model
    #model = DecisionTreeClassifier(max_depth=None, max_features=None, random_state=None )

    # fit the model
    #model.fit(X_df, y_df)

    #prediction
    pred = model.predict(X_df)

    #score = accuracy
    acc = model.score(X_df, y_df)

    #conf Matrix
    conf = confusion_matrix(y_df, pred)
    mat =  pd.DataFrame ((confusion_matrix(y_df, pred )),index = ['actual_dead','actual_survived'], columns =['pred_dead','pred_survived' ])
    rubric_df = pd.DataFrame([['True Negative', 'False positive'], ['False Negative', 'True Positive']], columns=mat.columns, index=mat.index)
    cf = rubric_df + ': ' + mat.values.astype(str)

    #assign the values
    tp = conf[1,1]
    fp =conf[0,1] 
    fn= conf[1,0]
    tn =conf[0,0]

    #calculate the rate
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    tnr = tn/(tn+fp)
    fnr = fn/(fn+tp)

    #classification report
    clas_rep =pd.DataFrame(classification_report(y_df, pred, output_dict=True)).T
    clas_rep.rename(index={'0': "dead", '1': "survived"}, inplace = True)
    print(f'''
    The accuracy for our model is {acc:.4%}
    The True Positive Rate is {tpr:.3%},    The False Positive Rate is {fpr:.3%},
    The True Negative Rate is {tnr:.3%},    The False Negative Rate is {fnr:.3%}
    ________________________________________________________________________________
    ''')
    print('''
    The positive is  'survived'
    Confusion Matrix
    ''')
    display(cf)
    print('''
    ________________________________________________________________________________
    
    Classification Report:
    ''')
    display(clas_rep)
   
