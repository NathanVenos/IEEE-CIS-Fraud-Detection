import pandas as pd

def score_classification(y_train, y_train_pred, y_test, y_test_pred):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import brier_score_loss
    from sklearn.metrics import f1_score
    from sklearn.metrics import log_loss
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import jaccard_score
    from sklearn.metrics import confusion_matrix
    
    scores = pd.DataFrame(data = np.array([[accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)],
                                          [balanced_accuracy_score(y_train, y_train_pred), balanced_accuracy_score(y_test, y_test_pred)],
                                          [precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)],
                                          [recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)],
                                          [f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)],
                                          [roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)],
                                          [brier_score_loss(y_train, y_train_pred), brier_score_loss(y_test, y_test_pred)],
                                          [log_loss(y_train, y_train_pred), log_loss(y_test, y_test_pred)],
                                          [jaccard_score(y_train, y_train_pred), jaccard_score(y_test, y_test_pred)]]),
                          index = ['Accuracy', 
                                   'Balanced_Accuracy', 
                                   'Precision', 
                                   'Recall', 
                                   'f1',
                                   'ROC_AUC',
                                   'Brier_Loss',
                                   'Log_Loss',
                                   'Jaccard'], 
                          columns = ['Train', 'Test'])
    print(scores)
    print(confusion_matrix(y_test, y_test_pred))
    
def downsample(df, target):
    from sklearn.utils import resample
    import pandas as pd

    is_0 =  df[target]==0 
    is_1 =  df[target]==1

    if is_0.sum() > is_1.sum():
        df_majority = df[is_0]
        df_minority = df[is_1]
    else:
        df_majority = df[is_1]
        df_minority = df[is_0]

    df_majority_downsampled = resample(df_majority, 
                                       replace=False,   
                                       n_samples=df_minority.shape[0],    
                                       random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled
 
def scaled_model_search(scalers, models, X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline

    best_score = 0
    
    for scaler in scalers:
        for model in models:
            pipe = Pipeline(steps=[('scaler', scaler),
                              ('classifier', model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
    print("The best model is {}, scaled by {}, with a test (accuracy) score of {}.".format(best_model, best_scaler, best_score)) 

import pandas as pd
def ngv_corr_var_df(df, corr_df, row_list=False):
    if row_list == False:
        row_list = [True for x in range(corr_df.shape[0])]
    corr_temp = corr_df.iloc[row_list].copy()
    # need to move index to a column to allow for .apply() to access column name for variance calculation
    corr_temp.reset_index(inplace=True)
    corr_temp.rename(columns={'index': 'feature'}, inplace=True)
    # calculating variance after dividing by mean to allow for even comparison
    corr_temp['variance'] = corr_temp['feature'].apply(lambda x: (df[x]/df[x].mean()).var())
    corr_temp.sort_values('variance', ascending=False, inplace=True)
    corr_temp.set_index('feature', inplace=True)
    return corr_temp

def ngv_reduce_feature_by_corr(df, feats, corr_filter = 0.95, orig_feats = [], keep_list = [], corr_df = []):
    # excluding anything already determined to be kept
    if keep_list == []:
        try_feats = feats
        orig_feats = feats
        corr_temp = df[try_feats].copy()
        corr_temp = corr_temp.corr()
    else:
        try_feats = list(set(feats).difference(set(keep_list)))
        corr_temp = corr_df
    row_list = []
    for row in corr_temp[try_feats].itertuples():
        bool_temp = [1 if (x > corr_filter) and (x < 1) else 0 for x in row[1:]]
        bool_temp = min(1, sum(bool_temp))
        bool_temp = bool(bool_temp)
        row_list.append(bool_temp)
        if bool_temp == False:
            keep_list.append(row[0])
    # controlling for duplicates
    keep_list = list(set(keep_list))  
    # if no more high correlations then return the results
    if sum(row_list) == 0:
        corr_temp = df[keep_list].corr()
        drop_list = list(set(orig_feats).difference(set(keep_list)))
        return keep_list, drop_list, ngv_corr_var_df(df, corr_temp)
    else:        
        # drop the feature with lowest variance and run again
        if try_feats == orig_feats:
            corr_temp = ngv_corr_var_df(df, corr_temp, row_list)
        drop_feat = corr_temp.index[-1]
        corr_temp = corr_temp.iloc[0:-1]
        corr_temp.drop(columns=drop_feat, inplace=True)
        next_feats = corr_temp.index.to_list()
        return ngv_reduce_feature_by_corr(df, next_feats, corr_filter, orig_feats, keep_list, corr_temp)