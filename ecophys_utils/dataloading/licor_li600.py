# LI-600 data loading functions
#-------------------------------
import glob
import pandas as pd

def load_li600(input_fn: str, silent: bool = True) -> pd.DataFrame:
    """
    Load a single LI-600 photosynthesis system file.

    Parameters
    ----------
    input_fn : str
        Path to the LI-600 CSV file.
    silent : bool, optional
        If False, print the file name being loaded. Default is True.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with parsed timestamps.
    """
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

def load_all_li600(path: str, pattern: str = '.csv', silent: bool = False) -> pd.DataFrame:
    """
    Load all LI-600 files from a directory.

    Parameters
    ----------
    path : str
        Directory path containing LI-600 files.
    pattern : str, optional
        File pattern to match. Default is '.csv'.
    silent : bool, optional
        If False, print loading progress. Default is False.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame from all files.
    """
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
    
def remove_obsolete_cols_li600(temp: pd.DataFrame, silent: bool = False) -> pd.DataFrame:
    """
    Remove obsolete columns from LI-600 DataFrame.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame from LI-600 file.
    silent : bool, optional
        If False, print action message. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with obsolete columns removed.
    """
    if (not silent):
        print('Removing obsolete columns')
    # Remove obsolete columns
    temp.drop('configAuthor', axis=1, inplace=True)
    temp.drop('configName', axis=1, inplace=True)
    temp.drop('configUpdatedAt', axis=1, inplace=True)
    temp.drop('flashID', axis=1, inplace=True)
    temp.drop('flashId', axis=1, inplace=True)
    temp.drop('lciSerNum', axis=1, inplace=True)
    temp.drop('lcpSerNum', axis=1, inplace=True)
    temp.drop('lcfSerNum', axis=1, inplace=True)
    temp.drop('lcrhSerNum', axis=1, inplace=True)
    temp.drop('version', axis=1, inplace=True)
    temp.drop('flash_type', axis=1, inplace=True)
    temp.drop('remark', axis=1, inplace=True)
    
    return(temp)