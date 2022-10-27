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
    
    #merging data
    tot_df = trans_df.merge(id_df, how='left', left_index=True, right_index=True)
    
    return tot_df