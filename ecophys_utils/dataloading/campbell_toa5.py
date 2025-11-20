# Campbell TOA5 data loading functions
#-----------------------------------
import pandas as pd

def load_campbell_toa5(input_fn: str) -> pd.DataFrame:
    """
    Load a Campbell TOA5 file.

    Parameters
    ----------
    input_fn : str
        File path to TOA5 file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with parsed timestamps.
    """
    df = pd.read_csv(input_fn,skiprows=[0,2,3], na_values=["NAN"])
    if(df.columns[0] != 'TIMESTAMP'):
        df = pd.read_csv(input_fn,skiprows=[0,1,3,4], na_values=["NAN"])
    df.rename(columns={'TIMESTAMP':'timestamp'}, inplace=True)
    #df['timestamp'] = pd.to_datetime( df.timestamp, format='%Y-%m-%d %H:%M:%S', errors="raise")
    df['timestamp'] = pd.to_datetime( df.timestamp, format='ISO8601', errors="raise")
    df.drop(columns=['RECORD'], inplace=True)
    return(df)