# CSV data loading functions
#---------------------------
import pandas as pd
from typing import Optional

def load_csv(fn: str, timestamp_format: str = '%Y-%m-%d %H:%M:%S', silent: bool = False) -> pd.DataFrame:
    """
    Load a single CSV file with timestamp parsing.

    Parameters
    ----------
    fn : str
        File path to CSV.
    timestamp_format : str, optional
        Timestamp format. Default is '%Y-%m-%d %H:%M:%S'.
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with parsed timestamps.
    """
    if not silent:
        print('  -', fn.split('/')[-1
    
    try:
        temp = pd.read_csv(fn, index_col=None)
        temp['timestamp'] = pd.to_datetime(temp['timestamp'], format=timestamp_format)
        return(temp)
    except FileNotFoundError:
        print(f'File not found: {fn}')
        raise
    except pd.errors.EmptyDataError:
        print(f'Empty CSV file: {fn}')
        raise
    except ValueError as e:
        print(f'Error parsing timestamps in {fn}: {str(e)}')
        raise

def load_all_csv(path: str, dataset: str = '.csv', timestamp_format: str = '%Y-%m-%d %H:%M:%S', silent: bool = False) -> pd.DataFrame:
    """
    Load all CSV files from a directory.

    Parameters
    ----------
    path : str
        Directory path.
    dataset : str, optional
        File extension or pattern. Default is '.csv'.
    timestamp_format : str, optional
        Timestamp format. Default is '%Y-%m-%d %H:%M:%S'.
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame from all files.
    """
    import os
    import glob
    
    if not silent:
        print('  - Loading ' + dataset + ' from ' + path)

    # Create the file list
    fn_list = sorted([f for f in glob.glob(os.path.join(path, '**'), recursive=True) if os.path.isfile(f) and (dataset in f.lower())])
    
    # For all files in the directory
    df_list = []
    for fn_i, fn in enumerate(fn_list):
        try:
            if not silent:  # Show progress for each file
                print( '\t{:<07}'.format(str(round(fn_i * 100 / len(fn_list), 4))) + "%\t" + fn.split('\\')[-2] + '\\' + fn.split('\\')[-1])
            # Load data
            temp = load_csv(fn, timestamp_format=timestamp_format, silent=True)
            df_list.append(temp)
        except Exception as e:
            print(f'Failed to load {fn}: {str(e)}')
            continue

    if not df_list:
        print(f'No valid CSV files found in {path}')
        raise ValueError('No valid CSV files found')

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Remove duplicates
    df = df.drop_duplicates()
    return(df)