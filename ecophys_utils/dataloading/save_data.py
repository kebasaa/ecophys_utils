def save_df(temp, output_path, output_fn, silent=True):
    import os
    import pandas as pd
    if(not silent):
        print('Saving data to', out_fn)
    # Ensure the directory exists
    os.makedirs(output_path, exist_ok=True)
    # Create full output file path
    out_fn = os.path.join(output_path, output_fn)
    # Save data
    temp.to_csv(out_fn, sep=',', index=False)
    if(not silent):
        print('Done...')
    pass
