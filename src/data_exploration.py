import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### plot_correlation_matrix_heat_map
#Returns a heatmap showing the N variables that are most correlated with a given column within a dataframe.

#Parameters:
#- df: a Pandas DataFrame to calculate correlation on
#- label: the variable to calculate correlation with
#- qty_fields: the number of variables (N) to display on the heatmap. ie, the dimension of the heatmap.

#This function uses the Python libraries [Pandas](https://pandas.pydata.org/docs/reference/index.html) (pd) and [Matplotlib](https://matplotlib.org/contents.html) (plt), both of which have been imported above.

def plot_correlation_matrix_heat_map(df,label,qty_fields=10):
    df = pd.concat([df[label],df.drop(label,axis=1)],axis=1)
    correlation_matrix = df.corr()
    index = correlation_matrix.sort_values(label, ascending=False).index
    correlation_matrix = correlation_matrix[index].sort_values(label,ascending=False)

    fig,ax = plt.subplots()
    fig.set_size_inches((10,10))
    sb.heatmap(correlation_matrix.iloc[:qty_fields,:qty_fields],annot=True,fmt='.2f',ax=ax)
    return(fig,ax)


### null_counts
#Returns a dataframe containing the number of null values in each column of a given dataframe.

#Parameters:
#- df: A DataFrame to check for null values.

#This function uses the Python libraries [Pandas](https://pandas.pydata.org/docs/reference/index.html) (pd), which has been imported above.

def null_counts(df):
    null_df = pd.DataFrame(df.isnull().sum(),columns=['null_count'])
    null_df['null_fraction'] = null_df['null_count'] / df.shape[0]
    null_df = null_df.sort_values('null_count',ascending=False)
    return null_df

def lj_basic_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    model = RandomForestClassifier(max_depth=5, random_state=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Score", model.score(X_test, y_test))
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    print("AUC", auc(fpr, tpr))
    return model

def lj_drop_low_correlation(X):
    vals = []
    for col in X.columns:
        corr = pearsonr(X[col], y)
        vals.append([col,corr[0], corr[1]])
    cols = []
    for col in sorted(vals, key=lambda x: x[1], reverse=True):
        if abs(col[1]) ==0:
            cols.append(col[0])
    X = X.drop(cols, axis=1)
    return X

def lj_drop_low_feature_importance(X, model):
    feats = []
    for vals in sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True):
        if vals[1] == 0 :
            feats.append(vals[0])
    X = X.drop(feats,axis=1)
    return X

def remove_false_cols(df_c):
    to_drop = []
    for col in df_c.columns:
        if col[-2:] == "_F":
            to_drop.append(col)
    df_c = df_c.drop(to_drop, axis=1)
    return df_c

def remove_false_cols(df_c):
    to_drop = []
    for col in df_c.columns:
        if col[-2:] == "_NotFound":
            to_drop.append(col)
    df_c = df_c.drop(to_drop, axis=1)
    return df_c
    

def lj_split_cat2(df):
    # split cols into categorical/numerical if already encoded
    cols = []
    for col in df.columns:
        #print(col)
        if len(df[col].unique() == 2) and df[col].unique().sum() == 1:
            cols.append(col)
    df_n = df.loc[:, ~df.columns.isin(cols)]
    df_c = df.loc[:, cols]
    return df_n, df_c

def remove_small_amts(df):
    df = df[df.transactionamt > 1].copy()
    return df

def lj_add_fp_correlation_features(X, model):
    X_pred = model.predict(X)
    X_all = X.copy()
    X_all["pred"] = X_pred
    X_all["isfraud"] = y
    X_all["fp"] = np.where((X_all["pred"]==1) & (X_all["isfraud"]==0), True, False)

    temp = X_all[X_all["fp"] == True].drop(['pred', 'isfraud', 'fp'], axis=1)
    
    crs = pd.DataFrame(temp.corr().unstack().sort_values(ascending=False)).reset_index()
    
    crs.columns = ["col1", "col2", "val"]
    
    crs = crs[crs["col1"] != crs["col2"]].copy()
    
    crs["comb"] = crs.apply(lambda x: "-".join(sorted([x["col1"], x["col2"]])), axis=1)
    
    crs = crs[["val", "comb"]].drop_duplicates()
    
    for col in crs[crs["val"] > .9].comb.values:
        col1, col2 = col.split("-")
        X[col] = X[col1] == X[col2]
        
    return X

from scipy.stats import pearsonr

# cleaning pipeline functions
def lj_get_df(df):
    
    
    # select target
    y = df["isfraud"]
    X = df.drop("isfraud", axis=1)
    
    
    df_n, df_c = split_num_cat(X)
    
    
    # filtering out high null count, low correlation columns
    nl_cts = null_counts(df)
    
    # fill Null by mean
    for col in df_n.columns:
        df_n[col] = df_n[col].fillna(df_n[col].mean())
        
    corrs = pd.DataFrame(columns=["col", "cor", "pval", "nl_cts"])
    for col in df_n.columns:
        cor, pval = pearsonr(df_n[col], y)
        corrs = corrs.append({"col":col, "cor":cor, "pval":pval, "nl_cts": nl_cts.loc[col]["null_fraction"]}, ignore_index=True)
    
    df_n = df_n.drop(corrs[(abs(corrs["nl_cts"]) > .7) & (abs(corrs["cor"] < .05))]["col"].values, axis=1)

    df_c = pd.get_dummies(df_c)   
    
    # auto feature selection
    #df_n = selectK(df_n, y, input_var="numerical", output_var="categorical")
    #df_c = selectK(pd.get_dummies(df_c), y, input_var="categorical", output_var="categorical")
    
    return(df_n, df_c, y)

from deco import synchronized

@synchronized
def lj_transform(df):
    
    df_n, df_c, y = lj_get_df(df)
    
    # concat
    X = pd.concat([df_n, df_c], axis=1)
    return (X, y)

def lj_fillna(df_n):
    for col in df_n.columns:
        df_n[col] = df_n[col].fillna(-1)
    return df_n
    
    
def filter_columns(df, df_n, df_c):
    
    from scipy.stats import zscore
    df2 = df[df["isfraud"] == 1].copy()
    cols = []
    for col in df_c.columns:
        temp = df_c[col].value_counts().to_dict()
        tmp2 = pd.DataFrame(df2[col].value_counts())
        tmp2 = tmp2.reset_index()
        tmp2.columns = ["nm", "total"]
        tmp2["pct"] = tmp2.apply(lambda x: (x.total / temp[x.nm]), axis=1)

        vals = [col + "_" + colval for colval in tmp2[zscore(tmp2["pct"]) > 2].nm.values]
        cols.append(vals)

    cols = sum(cols, [])

    def compare(val):
        answer = False
        cals = ["p_emaildomain", "r_emaildomain", "id_30", "id_31", "id_33", "deviceinfo"]
        for cal in cals:
            if cal in val: 
                answer = True
        return answer
    
    df_c = pd.get_dummies(df_c)
    filtered_cols = [col for col in df_c.columns if not compare(col) or col in cols]
    
    return df_c[filtered_cols]


def filter_by_corr(df, metric="pearson", threshold=.01):
    df_n, df_c = split_num_cat(df)
    from scipy.stats import pearsonr, kendalltau
    cols = []
    if metric == "pearson":
        func = pearsonr
    else:
        func = kendalltau
        
    for col in df_n.columns:
        cols.append([col, func(df_n[col], df["isfraud"])])
    
    filtered_cols = []
    for val in cols:
        if val[1][0] > threshold:
            filtered_cols.append(val[0])
    return df_n[filtered_cols]
    #print(sorted(cols, key=lambda x: x[1][0], reverse=True))


def remove_redundant(df_n):
    from itertools import combinations
    from scipy.stats import pearsonr
    from random import choice
    vals = []

    for combo in combinations(df_n.drop("isfraud", axis=1).columns, r=2):
        vals.append([combo[0] + "-" + combo[1], pearsonr(df_n[combo[0]], df_n[combo[1]])])
    sets = []
    for combo in vals:
        if combo[1][0] > .9:
            c1, c2 = combo[0].split("-")
            found = False
            for set_ in sets:
                if c1 in set_ or c2 in set_:
                    set_.append(c1)
                    set_.append(c2)
                    found = True
            if not found:
                sets.append([c1, c2])

    sets = [list(set(set_)) for set_ in sets]

    def remove_ret(set_, x):
        set_.remove(x)
        return set_

    to_remove = [remove_ret(set_, choice(set_)) for set_ in sets if len(set_) >0]
    to_remove = sum(to_remove, [])
    return df_n.drop(to_remove, axis=1)

def ngv_plot_corr_heatmap(dataframe, columns):
    # to allow for larger font sizes,
    # the correlations are being rounded and multiplied by 100
    corr_df = dataframe[columns].corr().round(2)*100
    
    # formatting
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize = (18, 14))
    
    # string conversion and NaN handling to keep labels concise in plot
    annots = corr_df.fillna(-999)
    annots = annots.astype(int)
    annots = annots.replace({-999: ''})
    annots = annots.astype(str)

    sns.heatmap(corr_df, center = 0, annot = annots, fmt="s")
