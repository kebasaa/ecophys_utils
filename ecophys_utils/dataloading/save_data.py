import os
import pandas as pd

def save_df(temp: pd.DataFrame, output_path: str, output_fn: str, silent: bool = False) -> None:
    """
    Save DataFrame to CSV file.

    Parameters
    ----------
    temp : pandas.DataFrame
        DataFrame to save.
    output_path : str
        Output directory path.
    output_fn : str
        Output file name.
    silent : bool, optional
        If False, print saving message. Default is False.
    """
    # Ensure the directory exists
    os.makedirs(output_path, exist_ok=True)
    # Create full output file path
    out_fn = os.path.join(output_path, output_fn)
    if(not silent):
        print('Saving data to', out_fn)
    # Save data
    temp.to_csv(out_fn, sep=',', index=False)
    #if(not silent):
    #    print('Done...')
    pass
