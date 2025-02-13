def load_li600(input_fn, silent=True):
    if (not silent):
        print('  -', input_fn.split('/')[-1])

    df = pd.read_csv(input_fn,skiprows=[0,2], na_values=[-9999,"NaN"])

    # Drop unused
    df.drop('Unnamed: 7', axis=1, inplace=True)
    df.drop('Unnamed: 8', axis=1, inplace=True)

    # Date & time
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y-%m-%d %H:%M:%S', errors="raise")
    # Move timestamp column to the front
    col = df.pop('timestamp')
    df.insert(0, col.name, col, allow_duplicates=True)
    df.drop('Date', axis=1, inplace=True)
    df.drop('Time', axis=1, inplace=True)
    
    return(df)

def load_all_li600(path, pattern='.csv', silent=False):
    if (not silent):
        print('Loading from ' + path)
        
    fn_list = sorted(glob.glob(path + '*' + pattern, recursive=True))
    
    # For all files in the directory
    df_list = []
    for fn_i, fn in enumerate(fn_list):
        if(silent == False): # % 20 to show every 20th file being loaded
            print( '\t{:<07}'.format(str(round(fn_i * 100 / len(fn_list), 4))) + "%\t" + fn.split('\\')[-1])
        # Load data
        temp = load_li600(fn)
        df_list.append(temp)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    return(df)