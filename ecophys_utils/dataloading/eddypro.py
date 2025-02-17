def load_eddypro(fn, silent=False):
    import pandas as pd
    if (not silent):
        print('  -', fn.split('/')[-1])
    if('biomet' not in fn):
        temp = pd.read_csv(fn, index_col=None, skiprows=[0,2], na_values=[-9999])
    else:
        temp = pd.read_csv(fn, index_col=None, skiprows=[1], na_values=[-9999])
    temp['timestamp'] = temp['date'] + ' ' + temp['time']
    temp['timestamp'] = pd.to_datetime(temp['timestamp'], format='%Y-%m-%d %H:%M')
    # Move timestamp to front
    col = temp.pop('timestamp')
    temp.insert(0, col.name, col, allow_duplicates=True)
    # Remove obsolete columns
    if('biomet' not in fn):
        remove_cols = ['filename','file_records','used_records','date','time','DOY']
    else:
        remove_cols = ['date','time','DOY']
    temp.drop(remove_cols, axis=1, inplace=True)
    return(temp)

def load_all_eddypro(path, dataset='full_output', silent=False):
    import os
    import glob
    import pandas as pd
    if (not silent):
        print('  - Loading ' + dataset + ' from ' + path)

    # Create the file list
    fn_list = sorted([f for f in glob.glob(os.path.join(path, '**'), recursive=True) if os.path.isfile(f) and (dataset in f.lower())])
    
    # For all files in the directory
    df_list = []
    for fn_i, fn in enumerate(fn_list):
        if(silent == False): # % 20 to show every 20th file being loaded
            print( '\t{:<07}'.format(str(round(fn_i * 100 / len(fn_list), 4))) + "%\t" + fn.split('\\')[-2] + '\\' + fn.split('\\')[-1])
        # Load data
        temp = load_eddypro(fn, silent=True)
        df_list.append(temp)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Remove duplicates
    df = df.drop_duplicates()
    return(df)