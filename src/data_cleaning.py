from deco import synchronized
import numpy as np

input_loc = '../data/raw/'

def standardize_col_names(df):
    '''
    Standardizes column names of a dataframe.
    It will remove white space, replace spaces with underscores, and eliminate special characters (including parenthesis and slashes).
    
    Parameters
    ----------
    
    df : Dataframe object
    
    Return Values
    -------------
    Dataframe with column names standardized.
    '''
    df.columns = (df.columns
                .str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('(', '')
                .str.replace(')', '')
                .str.replace('/','')
                .str.replace('\\',''))
    return df
    
        
def null_counts(df):
    '''
    Returns a dataframe containing the number of null values in each column of a given dataframe.
    
    Parameters
    ----------
    df : A DataFrame to check for null values.
    '''
    import pandas
    
    null_df = pandas.DataFrame(df.isnull().sum(), columns=['null_count'])
    null_df['null_fraction'] = null_df['null_count'] / df.shape[0]
    null_df = null_df.sort_values('null_count',ascending=False)
    return null_df


# LJ Functions
from deco import synchronized
# get, pivot and join data from 2 datasets
def lj_get_data(dataset="train"):
    import pandas as pd
    df1 = pd.read_csv(input_loc + '/' + dataset + '_transaction.csv')
    df2 = pd.read_csv(input_loc + '/' + dataset + '_identity.csv')
    df2 = df2.pivot(index="TransactionID", columns="variable", values="value")
    df = df1.join(df2, on="TransactionID", how="left", lsuffix="df1", rsuffix="df2")
    df = df.set_index("TransactionID")   
    return df


def lj_drop_dupes(df):
    #drop duplicate rows, columns
    df = df.drop_duplicates()
    df = df.loc[:,~df.columns.duplicated()]
    
    # remove null target rows
    df = df[df["isfraud"].notna()].copy()
    
    # removing all null count column
    df = df.drop("v340", axis=1)
    
    return df

def lj_object_to_float(df):
    # converts all object columns that contain all numerical data to float
    tmp = df.mode()
    for col in tmp.columns:
        if df[col].dtype == "object":
            try: 
                float(tmp[col].iloc[0])
                df[col] = df[col].astype(float)
            except:
                pass
    return df

def lj_bin_fields(df):
    df["r_emaildomain"] = np.where(df["r_emaildomain"].apply(lambda x: ".com" in str(x) or ".net" in str(x)), True, False)
    df["p_emaildomain"] = np.where(df["p_emaildomain"].apply(lambda x: ".com" in str(x) or ".net" in str(x)), True, False)
    df.deviceinfo = df.deviceinfo.apply(lambda x: x if x in ["Windows", "iOS Device", "MacOS"] else "Other")

    def bin_os(x):
        x = str(x)
        if "Windows" in x: return "windows"
        if "Android" in x: return "android"
        if "OS X" in x: return "osx"
        if "iOS" in x: return "ios"
        else: return "other"
    df.id_30 = df.id_30.apply(bin_os)

    return df


@synchronized
def lj_clean(funcs=[]):
    # calls all cleaning functions after getting data, with multiprocessing
    df = lj_get_data()
    
    if len(funcs) == 0:
        funcs = [standardize_col_names, lj_drop_dupes, lj_object_to_float] #, lj_bin_fields]
        
    for func in funcs:
        df = func(df)
        
    return df

# ngv added functions
def ngv_reshape_id_data(df, index_col='transactionid', pivot_col = 'variable', values_col = 'value'):
    '''
    Returns a dataframe that is pivoted version of the input identity dataframe.
    
    Parameters
    ----------
    df : An identity dataframe (train, test, etc.)
    index_col : The index of df
    pivot_col: The column to pivot in df
    values_col : The column with the pivot column's values
    
    Return Values
    -------------
    Dataframe with pivoted column of data into multiple columns of data
    '''
    import pandas
    
    rs_df = df.pivot(index=index_col, columns=pivot_col, values=values_col)
    rs_df = rs_df.apply(pandas.to_numeric, errors='ignore', downcast='integer')
    return rs_df

def ngv_reencode_match_status(df, col='id_34'):
    '''
    Returns a dataframe where certain columns are re-encoded such that
    the "match status:" string has been stripped from the column values
    and the remaining digits are converted to integers and populate the columns.
    Digits are also incremented by 1 so that they start at 0 instead of -1.
    
    Parameters
    ----------
    df : A DataFrame to re-encode
    col: A column name specifying which column to re-encode
        
    Return Values
    -------------
    Dataframe with specified column values re-encoded
    '''
    import pandas
    
    df[col] = df[col].str.strip('match_status:')
    df[col] = pandas.to_numeric(df[col], downcast='integer')
    df[col] += 1
    return df

def ngv_reencode_mapping(df, cols, map_dict):
    '''
    Returns a dataframe where certain columns are re-encoded such that
    values are re-assigned based on the provided mapping of values.
    
    Parameters
    ----------
    df : A DataFrame to re-encode
    cols: A list of column names on which to map the map_dict
    map_dict: A dictionary containing the mapping of old to new values among the cols
        
    Return Values
    -------------
    Dataframe with specified column values re-encoded
    '''
    import pandas
    
    for col in cols:
        df[col] = df[col].map(map_dict)
    return df

def ngv_label_encode(label_col_list, df):
    label_dicts = {}
    # gets unique values in reverse order by frequency
    for cat in label_col_list:
        temp_keys = list(df[cat].value_counts().index)[::-1]
        temp_values = range(len(temp_keys))
        temp_dict = dict(zip(temp_keys, temp_values))
        label_dicts.update({cat: temp_dict})
    # performs the encoding
        df[cat] = df[cat].map(temp_dict)
    return df

def ngv_ord_encode_fit(label_col_list, df):
    import pandas as pd
    import numpy as np
    encoder_dicts = {}
    # gets unique values
    for cat in label_col_list:
        codes, uniques = pd.factorize(df[cat])
        encodes = np.unique(codes)[1:]
        encoder = dict(zip(uniques, encodes))
    # saves the encoding
        encoder_dicts.update({cat: encoder})
    return encoder_dicts

def ngv_ord_encode_transform(encoder_dicts, df, encode_na=False):
    import pandas as pd
    for cat in encoder_dicts:
        encoder = encoder_dicts[cat]
        if encode_na:
            # makes Nan's a category
            na_val = max(encoder.values) + 1
        else:
            # encodes Nan's as -1 which lgbm interprets as a Nan
            na_val = -1
        df[cat] = pd.to_numeric(df[cat].map(encoder),
                                errors="coerce").fillna(na_val)
    return df

def ngv_freq_encode(freq_col_list, df):
    # creating the list of mappings in case you want to return it for inspection
    freq_dicts = {}
    # gets frequency counts
    for cat in freq_col_list:
        temp_dict = df[cat].value_counts(normalize=True, dropna=True).to_dict()
        freq_dicts.update({cat: temp_dict})
    # performs the encoding
        df[cat] = df[cat].map(temp_dict)
    return df


import re

def ngv_strip_os_name(os_id):
    # extracts the OS name from column id_30
    try:
        num_ix = re.search(r"\d", os_id)
        if (num_ix != None):
            os_name = os_id[:num_ix.start()-1]
        else:
            os_name = os_id
        return os_name
    except:
        return os_id

def ngv_strip_os_num(os_id):
    # extracts the OS version number from column id_30
    try:
        num_ix = re.search(r"\d", os_id)
        if (num_ix != None):
            os_name = os_id[num_ix.start():]
        else:
            os_name = np.nan
        return os_name
    except:
        return os_id


def ngv_id_31_binning(item, bin_list, many=True):
    # bins/groups the entries based on a passed list of browser type strings
    if item is np.nan:
        return item
    else:        
        for ibin in bin_list:
            if ibin in item.lower():
                return ibin
                break
        if many == True:
            return item
        else:
            return 'other'

def ngv_resolution_prod(res):
    # calculates the product of the screen resolution values
    if res is np.nan:
        return res
    else:
        txt = res.split('x')
        txt = np.array(txt, dtype='int')
        return txt.sum()

def ngv_resolution_ratio(res, dec=1):
    # creates a rounded ratio based on the screen resolution values
    if res is np.nan:
        return res
    else:
        txt = res.split('x')
        txt = np.array(txt, dtype='int')
        max_txt = txt.max()
        min_txt = txt.min()
        ratio = max_txt / min_txt
        return round(ratio, dec)

def ngv_over_two_decimal_amt(col, df):
    # populates a new column with 1 if transactionamt has
    # three or more decimal points otherwise 0
    # intended to identify foreign currency conversion
    new_col = col+'_long_dec'
    df[new_col] = df[col].apply(lambda x: len(str(x).split('.')[1]))
    df[new_col] = df[new_col].apply(lambda x: 1 if x > 2 else 0)
    return df

def ngv_initial_data_setup(trans_path, id_path):
    '''
    Returns a dataframe that uploads, cleans, and merges
    transaction and identity data (i.e. train or test) per ngv's specifications.
    
    Parameters
    ----------
    trans_path : A path to the transaction data
    id_path: A path to the identity data
        
    Return Values
    -------------
    Dataframe with cleaned and merged transaction and identity data.
    '''
    import pandas
    
    # loading data
    trans_df = pandas.read_csv(trans_path)
    id_df = pandas.read_csv(id_path)
    
    # standardizing column names
    trans_df = standardize_col_names(trans_df)
    id_df = standardize_col_names(id_df)
    
    # reshaping and reindexing data
    id_df = ngv_reshape_id_data(id_df)
    trans_df.set_index('transactionid', drop=True, inplace=True)

    # dropping column with all nulls and duplicates
    trans_df.drop(['v340'], axis=1, inplace=True)
    trans_df = trans_df.loc[:, ~trans_df.columns.duplicated()]
    
    # re-encoding mappings
    tf_id_bools = ['id_35', 'id_36', 'id_37', 'id_38']
    tf_trans_bools = ['m1', 'm2', 'm3', 'm5', 'm6', 'm7', 'm8', 'm9']
    tf_map = {'T': 1, 'F': 0}
    found_bools = ['id_12', 'id_16', 'id_27', 'id_28', 'id_29']
    found_map = {'Found': 1, 'NotFound': 0, 'New': 0}
    device_bools =['DeviceType']
    device_map = {'desktop': 1, 'mobile': 0}
   
    #re-encodings
    id_df = ngv_reencode_match_status(id_df)
    trans_df = ngv_reencode_mapping(trans_df, tf_trans_bools, tf_map)
    id_df = ngv_reencode_mapping(id_df, tf_id_bools, tf_map)
    id_df = ngv_reencode_mapping(id_df, found_bools, found_map)
    id_df = ngv_reencode_mapping(id_df, device_bools, device_map)
    

    # pre-processing various categorical columns with many categories
    # to help reduce dimensionality by binning/grouping them in relevant ways
    # useful lists being generated for looping/binning
    id_31_bin_list = ['chrome', 'safari', 'firefox',
                      'opera', 'edge', 'samsung browser',
                      'ie', 'google', 'android']
    trans_label_cats = ['productcd', 'card4', 'card6', 'm4', 'p_emaildomain', 'r_emaildomain']
    id_label_cats = ['id_15', 'id_23', 'id_30_name', 'id_31_bin_few']
    # will frequency encode high dimensional categories as well as those with time sensitive details such as version numbers
    freq_col_list = ['DeviceInfo', 'DeviceInfo_trunc', 'DeviceInfo_alpha_trunc', 'id_30', 'id_31', 'id_31_bin_many', 'id_33']

    # grouping and pre-processing categoricals in advance of encoding
    id_df['id_30_name'] = id_df['id_30'].apply(lambda x: ngv_strip_os_name(x))
    # decided not to create this feature
    # id_df['id_30_num'] = id_df['id_30'].apply(lambda x: ngv_strip_os_num(x))
    id_df['id_31_bin_many'] = id_df['id_31'].apply(lambda x: ngv_id_31_binning(x, id_31_bin_list, many=True))
    id_df['id_31_bin_few'] = id_df['id_31'].apply(lambda x: ngv_id_31_binning(x, id_31_bin_list, many=False))
    id_df['id_33_prod'] = id_df['id_33'].apply(lambda x: ngv_resolution_prod(x))
    id_df['id_33_ratio'] = id_df['id_33'].apply(lambda x: ngv_resolution_ratio(x, dec=1))
    id_df['DeviceInfo_trunc'] = id_df['DeviceInfo']
    id_df['DeviceInfo_trunc'] = id_df['DeviceInfo_trunc'].apply(lambda x: x[:min(len(x),6)] if x is not np.nan else x)
    id_df['DeviceInfo_alpha_trunc'] = id_df['DeviceInfo']
    id_df['DeviceInfo_alpha_trunc'] = id_df['DeviceInfo_alpha_trunc'].apply(lambda x: ''.join([i for i in x.lower() if i.isalpha()])
                                                                                      if x is not np.nan else x)
    id_df['DeviceInfo_alpha_trunc'] = id_df['DeviceInfo_alpha_trunc'].apply(lambda x: x[:min(len(x),6)] if x is not np.nan else x)

    # encoding the pre-processed categorical data
    trans_df = ngv_label_encode(trans_label_cats, trans_df)
    trans_df = ngv_over_two_decimal_amt('transactionamt', trans_df)

    id_df = ngv_label_encode(id_label_cats, id_df)
    id_df = ngv_freq_encode(freq_col_list, id_df)

    #merging data
    tot_df = trans_df.merge(id_df, how='left', left_index=True, right_index=True)
    
    return tot_df



def ngv_train_test_data_processor(tr_trans_path, tr_id_path, ts_trans_path, ts_id_path):
    '''
    Returns a dataframe that uploads, cleans, merges, selects, and encodes
    transaction and identity data (i.e. train or test) per ngv's specifications.
    
    Parameters
    ----------
    tr_trans_path : A path to the transaction data
    tr_id_path: A path to the identity data
    ts_trans_path : A path to the transaction data
    ts_id_path: A path to the identity data
        
    Return Values
    -------------
    Training Dataframe with data ready for modeling.
    Testing Dataframe with data ready for modeling.
    '''
    import pandas
    
    # loading data
    tr_trans_df = pandas.read_csv(tr_trans_path)
    tr_id_df = pandas.read_csv(tr_id_path)
    # now testing
    ts_trans_df = pandas.read_csv(ts_trans_path)
    ts_id_df = pandas.read_csv(ts_id_path)
    
    # standardizing column names
    tr_trans_df = standardize_col_names(tr_trans_df)
    tr_id_df = standardize_col_names(tr_id_df)
    # now testing
    ts_trans_df = standardize_col_names(ts_trans_df)
    ts_id_df = standardize_col_names(ts_id_df)
    
    # reshaping and reindexing data
    tr_id_df = ngv_reshape_id_data(tr_id_df)
    tr_trans_df.set_index('transactionid', drop=True, inplace=True)
    # now testing
    ts_id_df = ngv_reshape_id_data(ts_id_df)
    ts_trans_df.set_index('transactionid', drop=True, inplace=True)

    # dropping column with all nulls and duplicates
    tr_trans_df.drop(['v340'], axis=1, inplace=True)
    tr_trans_df = tr_trans_df.loc[:, ~tr_trans_df.columns.duplicated()]
    # now testing
    ts_trans_df.drop(['v340'], axis=1, inplace=True)
    ts_trans_df = ts_trans_df.loc[:, ~ts_trans_df.columns.duplicated()]
    
    # default re-encoding mappings
    tf_id_bools = ['id_35', 'id_36', 'id_37', 'id_38']
    tf_trans_bools = ['m1', 'm2', 'm3', 'm5', 'm6', 'm7', 'm8', 'm9']
    tf_map = {'T': 1, 'F': 0}
    found_bools = ['id_12', 'id_16', 'id_27', 'id_28', 'id_29']
    found_map = {'Found': 2, 'NotFound': 1, 'New': 0}
    device_bools =['DeviceType']
    device_map = {'desktop': 1, 'mobile': 0}
   
    # default re-encodings
    tr_id_df = ngv_reencode_match_status(tr_id_df)
    tr_trans_df = ngv_reencode_mapping(tr_trans_df, tf_trans_bools, tf_map)
    tr_id_df = ngv_reencode_mapping(tr_id_df, tf_id_bools, tf_map)
    tr_id_df = ngv_reencode_mapping(tr_id_df, found_bools, found_map)
    tr_id_df = ngv_reencode_mapping(tr_id_df, device_bools, device_map)
    # now testing
    ts_id_df = ngv_reencode_match_status(ts_id_df)
    ts_trans_df = ngv_reencode_mapping(ts_trans_df, tf_trans_bools, tf_map)
    ts_id_df = ngv_reencode_mapping(ts_id_df, tf_id_bools, tf_map)
    ts_id_df = ngv_reencode_mapping(ts_id_df, found_bools, found_map)
    ts_id_df = ngv_reencode_mapping(ts_id_df, device_bools, device_map)
    

    # pre-processing various categorical columns with many categories
    # to help reduce dimensionality by binning/grouping them in relevant ways
    # useful lists being generated for looping/binning
    id_31_bin_list = ['chrome', 'safari', 'firefox',
                      'opera', 'edge', 'samsung browser',
                      'ie', 'google', 'android']

    # grouping and pre-processing categoricals in advance of encoding
    tr_id_df['id_30_name'] = tr_id_df['id_30'].apply(lambda x: ngv_strip_os_name(x))
    tr_id_df['id_31_bin_many'] = tr_id_df['id_31'].apply(lambda x: ngv_id_31_binning(x, id_31_bin_list, many=True))
    tr_id_df['id_31_bin_few'] = tr_id_df['id_31'].apply(lambda x: ngv_id_31_binning(x, id_31_bin_list, many=False))
    tr_id_df['id_33_prod'] = tr_id_df['id_33'].apply(lambda x: ngv_resolution_prod(x))
    tr_id_df['id_33_ratio'] = tr_id_df['id_33'].apply(lambda x: ngv_resolution_ratio(x, dec=1))
    tr_id_df['DeviceInfo_trunc'] = tr_id_df['DeviceInfo']
    tr_id_df['DeviceInfo_trunc'] = tr_id_df['DeviceInfo_trunc'].apply(lambda x: x[:min(len(x),6)] if x is not np.nan else x)
    tr_id_df['DeviceInfo_alpha_trunc'] = tr_id_df['DeviceInfo']
    tr_id_df['DeviceInfo_alpha_trunc'] = tr_id_df['DeviceInfo_alpha_trunc'].apply(lambda x: ''.join([i for i in x.lower() if i.isalpha()])
                                                                                      if x is not np.nan else x)
    tr_id_df['DeviceInfo_alpha_trunc'] = tr_id_df['DeviceInfo_alpha_trunc'].apply(lambda x: x[:min(len(x),6)] if x is not np.nan else x)
    tr_trans_df = ngv_over_two_decimal_amt('transactionamt', tr_trans_df)
    # now testing
    ts_id_df['id_30_name'] = ts_id_df['id_30'].apply(lambda x: ngv_strip_os_name(x))
    ts_id_df['id_31_bin_many'] = ts_id_df['id_31'].apply(lambda x: ngv_id_31_binning(x, id_31_bin_list, many=True))
    ts_id_df['id_31_bin_few'] = ts_id_df['id_31'].apply(lambda x: ngv_id_31_binning(x, id_31_bin_list, many=False))
    ts_id_df['id_33_prod'] = ts_id_df['id_33'].apply(lambda x: ngv_resolution_prod(x))
    ts_id_df['id_33_ratio'] = ts_id_df['id_33'].apply(lambda x: ngv_resolution_ratio(x, dec=1))
    ts_id_df['DeviceInfo_trunc'] = ts_id_df['DeviceInfo']
    ts_id_df['DeviceInfo_trunc'] = ts_id_df['DeviceInfo_trunc'].apply(lambda x: x[:min(len(x),6)] if x is not np.nan else x)
    ts_id_df['DeviceInfo_alpha_trunc'] = ts_id_df['DeviceInfo']
    ts_id_df['DeviceInfo_alpha_trunc'] = ts_id_df['DeviceInfo_alpha_trunc'].apply(lambda x: ''.join([i for i in x.lower() if i.isalpha()])
                                                                                      if x is not np.nan else x)
    ts_id_df['DeviceInfo_alpha_trunc'] = ts_id_df['DeviceInfo_alpha_trunc'].apply(lambda x: x[:min(len(x),6)] if x is not np.nan else x)
    ts_trans_df = ngv_over_two_decimal_amt('transactionamt', ts_trans_df)
#     return tr_trans_df, tr_id_df, ts_trans_df, ts_id_df

    #merging data
    tr_tot_df = tr_trans_df.merge(tr_id_df, how='left', left_index=True, right_index=True)
    ts_tot_df = ts_trans_df.merge(ts_id_df, how='left', left_index=True, right_index=True)
    
    return tr_tot_df, ts_tot_df

def ngv_float_to_int(col):
    import pandas as pd
    import numpy as np
    if (col.astype(np.int64, errors='ignore') == col).all():
        return pd.to_numeric(col, downcast='integer')
    else:
        return pd.to_numeric(col, downcast='float')

def split_num_cat(df):
    cols = []
    for col in df.columns:
        if df[col].dtype in ["object", "bool", "int"]:
            cols.append(col)
    df_n = df.loc[:, ~df.columns.isin(cols)]
    df_c = df.loc[:, cols]
    return df_n, df_c


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