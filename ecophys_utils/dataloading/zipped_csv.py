import zipfile
import pandas as pd
    
def load_all_zip(path: str, silent: bool = False) -> pd.DataFrame:
    """
    Load all CSV files from within a ZIP archive.

    Parameters
    ----------
    path : str
        Path to the ZIP file.
    silent : bool, optional
        If False, print loading progress. Default is False.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame from all CSV files in the ZIP.
    """

    # Open ZIP file and list all CSVs
    with zipfile.ZipFile(path, "r") as z:
        fn_list = [f for f in z.namelist() if f.endswith(".csv")]

        if (not silent):
            print('Loading from ' + path + '\t(' + str(len(fn_list)) + ' files)')

        df_list = []
        # Read all CSV files into a dictionary of DataFrames
        #df_list = {file: pd.read_csv(z.open(file), index_col=None) for file in csv_files}
        for fn_i, fn in enumerate(fn_list): #file in csv_files:
            if((not silent) & (fn_i % 100 == 0)): # % 100 to show every 100th file being loaded
                print( '\t{:<07}'.format(str(round(fn_i * 100 / len(fn_list), 4))) + "%\t" + fn.split('\\')[-1])
            # Load data
            temp = pd.read_csv(z.open(fn), index_col=None)
            temp['timestamp'] = pd.to_datetime(temp['timestamp'], format='%Y-%m-%d %H:%M:%S')
            df_list.append(temp)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    return(df)