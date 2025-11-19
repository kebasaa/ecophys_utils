def load_csv(fn, timestamp_format='%Y-%m-%d %H:%M:%S', silent=False):
    import pandas as pd
    if (not silent):
        print('  -', fn.split('/')[-1])
    temp = pd.read_csv(fn, index_col=None)
    temp['timestamp'] = pd.to_datetime(temp['timestamp'], format=timestamp_format)
    return(temp)

def load_all_csv(path, dataset='.csv', timestamp_format='%Y-%m-%d %H:%M:%S', silent=False):
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
        temp = load_csv(fn, timestamp_format=timestamp_format, silent=True)
        df_list.append(temp)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Remove duplicates
    df = df.drop_duplicates()
    return(df)