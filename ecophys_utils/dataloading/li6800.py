def load_li6800(input_fn, silent=True):
    import os
    import pandas as pd
    from ..misc.misc import sanitize_column_names
    
    if (not silent):
        print('  -', input_fn.split('/')[-1])

    from io import StringIO

    # Read file line by line
    with open(input_fn, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    # Prepare for storage of data
    header = None
    data_lines = []
    file_date = None

    capture_data = False
    capture_metadata = False

    # Loop through each line in the file
    for i, line in enumerate(lines):
        # From the metadata, extract the file date
        if capture_metadata:
            file_date = line[12:31]
            capture_metadata = False
        # Detect the beginning of the metadata
        if line.startswith('"OPEN'):
            capture_metadata = True

        # Begin capturing data lines
        if line.startswith('$STARTOFDATA$'):
            capture_data = True
        elif capture_data and not line.startswith('"'):
            # Append non-header (non-quoted) lines once data capturing has started
            data_lines.append(line)
        elif line.startswith('"Obs"'):
            # Get the header
            header = line.replace('"', '').split('\t')
            header = sanitize_column_names(header)

    # Join & convert into df
    data_str = "\n".join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep='\t', header=None, names=header)

    # Add file date & convert to timestamp
    df['timestamp'] = pd.to_datetime(file_date + ' ' + df['HHMMSS'].astype(str), format='%b %d %Y %H:%M:%S')

    # Extract the file name
    filename = os.path.basename(input_fn)
    df['filename'] = filename

    # Move the timestamp column to the front
    col = df.pop('filename')
    df.insert(0, col.name, col, allow_duplicates=True)
    col = df.pop('timestamp')
    df.insert(0, col.name, col, allow_duplicates=True)

    # Remove the HHMMSS column (obsolete after creating the timestamp).
    df.drop(columns='HHMMSS', inplace=True)
    
    return(df)

def load_all_li6800(path, silent=False):
    import os
    import glob
    import pandas as pd
    if (not silent):
        print('Loading from ' + path)

    # Create the file list
    fn_list = sorted([f for f in glob.glob(os.path.join(path, '**'), recursive=True) if os.path.isfile(f) and not f.lower().endswith(('.xls', '.xls$'))])
    
    # For all files in the directory
    df_list = []
    for fn_i, fn in enumerate(fn_list):
        if(silent == False): # % 20 to show every 20th file being loaded
            print( '\t{:<07}'.format(str(round(fn_i * 100 / len(fn_list), 4))) + "%\t" + fn.split('\\')[-1])
        # Load data
        temp = load_li6800(fn)
        df_list.append(temp)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Remove duplicates
    df = df.drop_duplicates()
    return(df)