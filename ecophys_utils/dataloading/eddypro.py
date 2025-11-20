# EddyPro data loading functions
#-------------------------------
import pandas as pd
from typing import Optional

def load_eddypro(fn: str, silent: bool = False) -> pd.DataFrame:
    """
    Load a single EddyPro file.

    Parameters
    ----------
    fn : str
        File path to EddyPro file.
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with parsed timestamps.
    """
    if (not silent):
        print('  -', fn.split('/')[-1])
    try:
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
    except FileNotFoundError:
        print(f'File not found: {fn}')
        raise
    except pd.errors.EmptyDataError:
        print(f'Empty EddyPro file: {fn}')
        raise
    except ValueError as e:
        print(f'Error parsing timestamps in {fn}: {str(e)}')
        raise

def load_all_eddypro(path: str, dataset: str = 'full_output', silent: bool = False) -> pd.DataFrame:
    """
    Load all EddyPro files from a directory.

    Parameters
    ----------
    path : str
        Directory path.
    dataset : str, optional
        File pattern. Default is 'full_output'.
    silent : bool, optional
        Whether to suppress output. Default is False.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame from all files.
    """
    import os
    import glob
    if (not silent):
        print('  - Loading ' + dataset + ' from ' + path)

    # Create the file list
    fn_list = sorted([f for f in glob.glob(os.path.join(path, '**'), recursive=True) if os.path.isfile(f) and (dataset in f.lower())])
    
    # For all files in the directory
    df_list = []
    for fn_i, fn in enumerate(fn_list):
        if(silent == False): # % 20 to show every 20th file being loaded
            print( '\t{:<07}'.format(str(round(fn_i * 100 / len(fn_list), 4))) + "%\t" + fn.split('\\')[-2] + '\\' + fn.split('\\')[-1])
        try:
            # Load data
            temp = load_eddypro(fn, silent=True)
            df_list.append(temp)
        except Exception as e:
            print(f'Failed to load {fn}: {str(e)}')
            continue

    if not df_list:
        print(f'No valid EddyPro files found in {path}')
        raise ValueError('No valid EddyPro files found')

    df = pd.concat(df_list, axis=0, ignore_index=True)

    # Remove duplicates
    df = df.drop_duplicates()
    return(df)