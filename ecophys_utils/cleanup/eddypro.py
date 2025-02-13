# Removes flagged data by making it NAN, based on a condition flag
def flagged_data_removal_ep(temp, col, flag, silent=False):
    temp = temp.copy()

    # Stats
    n_bad = len(temp.loc[flag].index)
    n     = len(temp.loc[~temp[col].isna()].index)

    if(not silent and not n):
        print('  - WARNING: Removing', col, 'data failed: All NA')
        return(temp[col])
    
    if not silent:
        print('  - Removing', str(np.round(n_bad/n*100, 2)) + '% flagged', col, 'data')

    # Remove bad data
    temp.loc[flag, col] = np.nan
    
    return(temp[col])