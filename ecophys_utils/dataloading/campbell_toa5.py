# Read a TOA5 input file
def load_toa5(input_fn):
    import pandas as pd
    df = pd.read_csv(input_fn,skiprows=[0,2,3], na_values=["NAN"])
    if(df.columns[0] != 'TIMESTAMP'):
        df = pd.read_csv(input_fn,skiprows=[0,1,3,4], na_values=["NAN"])
    df.rename(columns={'TIMESTAMP':'timestamp'}, inplace=True)
    #df['timestamp'] = pd.to_datetime( df.timestamp, format='%Y-%m-%d %H:%M:%S', errors="raise")
    df['timestamp'] = pd.to_datetime( df.timestamp, format='ISO8601', errors="raise")
    df.drop(columns=['RECORD'], inplace=True)
    return(df)