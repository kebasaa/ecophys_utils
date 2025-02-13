![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# ecophys_utils: Python utilities library for ecophysiology tasks

Library of support functions for Ecophysiology (Eddy Covariance, flux calculations, etc.). Functions are organised by files

## Documentation (Available functions)

1. **dataloading:** Load data from a range of devices. Contains the following functions:
    - _Eddypro:_ [Eddypro (LI-COR Biosciences)](https://www.licor.com/support/EddyPro/software.html) outputs
	    - `load_eddypro(fn, silent=False)`: Load a single file, supports "biomet" and "full_output"
		- `load_all_eddypro(path, dataset='full_output', silent=False)`: Load a folder of files, supports "biomet" and "full_output"
	- _LI-600_: [LI-COR LI-600 Porometer/ Fluorometer](https://www.licor.com/products/LI-600) outputs
	    - `load_li600(input_fn, silent=True)`: Load a single file
		- `load_all_li600(path, pattern='.csv', silent=False)`: Load a folder of files
	- _LI-6400_: [LI-COR LI-6400 Portable Photosynthesis System](https://www.licor.com/support/LI-6400/topics/system-description.html) outputs
	    - `load_li6400(input_fn, silent=True)`: Load a single file
		- `load_all_li6400(path, silent=False)`: Load a folder of files
	- _Zipped CSVs:_ All CSVs contained in a zip file
	    - `load_all_zip(path, silent=False)`
2. **meteo:** Meteorology functions
    - `calculate_last_precipitation(df, timestamp_col='timestamp', precipitation_col='P_1_1_1', max_gap_in_event_h=12)`: Calculates time since last precipitation event, and the summed amount of the event
3. **misc:** Miscellaneous functions
    - `upsample_interpolate_df(temp, freq='1min', interpolation_limit=30)`: Upsamples a df, e.g. from 30min to 1min, and interpolates the number of data points given in "interpolation_limit"
    - `sanitize_column_names(header)`: Cleans column names by removing parentheses, underscores and other special characters. Usage: `df.columns = sanitize_column_names(df.columns)`
4. **cleanup:** Cleanup functions, e.g. permitting the removal of flagged data
    - `flagged_data_removal_ep(temp, col, flag, silent=False)`: Cleans up a data column (col) by setting values to NAN if the flag applies, useful here for Eddypro flags. Usage example: `df['H'] = flagged_data_removal_ep(df, 'H', (df['qc_H'] >= 2))`

	
## Usage

There are a few options on how to use this package. We assume that you are using an anaconda environment called "your_env", you are using Windows and your user name is "my_user". Modify as necessary:

1. Add it to _sys.path_ temporarily. In any script, add the following at the top (replace "my_user" with your Windows user name):

    ```python
    import sys
    sys.path.append("C:/Users/my_user/Documents/Github/ecophys_utils/")
    from ecophys_utils import dataloading
    ```

2. Add it to PYTHONPATH permanently. This allows python to always find it and be easily imported
    - Locate the site-packages directory for your my_env environment. This will output a path like C:\Users\my_user\anaconda3\envs\my_env\Lib\site-packages\:
        ```bash
        conda activate my_env
        python -c "import site; print(site.getsitepackages())"
        ```

    - Create a file named ecophys_utils.pth inside the folder found above, and add the following line to it:
	
        ```
        C:\Users\my_user\Documents\Github\ecophys_utils\
        ```

    - Now, you can simply use the library:

        ```python
        from ecophysutils import dataloading
        ```
	
3. Install it as a package:
    ```
    conda activate my_env
    pip install -e C:\Users\my_user\Documents\Github\ecopyhs_utils\
    ```

## How to Cite (Update later once useful)

Muller (2025). *ecophys_utils: Python utilities library for ecophysiology tasks*

## License

This software is distributed under the GNU GPL version 3

