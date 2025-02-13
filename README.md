![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)


[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# ecophys_utils: Python utilities library for ecophysiology tasks

Library of support functions for Ecophysiology (Eddy Covariance, flux calculations, etc.). Functions are organised by files

## Description of available functions

1. **dataloading:** Functions for loading data from a range of devices. Contains the following functions:
    - _load_eddypro_: Functions for loading output files of [Eddypro (LI-COR Biosciences)](https://www.licor.com/support/EddyPro/software.html)
	- _load_li600_: Functions for loading output files of the [LI-COR LI-600 Porometer/ Fluorometer](https://www.licor.com/products/LI-600)
	- _load_li6400_: Functions for loading output files of the [LI-COR LI-6400 Portable Photosynthesis System](https://www.licor.com/support/LI-6400/topics/system-description.html)
	
## Usage

There are a few options on how to use this package. We assume that you are using an anaconda environment called "your_env", you are using Windows and your user name is "my_user". Modify as necessary:

1. Add it to _sys.path_ temporarily. In any script, add the following at the top (replace "my_user" with your Windows user name):

```python
import sys
sys.path.append(r"C:\Users\my_user\Documents\Github\")
from ecophys_utils import dataloading
```

2. Add it to PYTHONPATH permanently. This allows python to always find it and be easily imported
    - Locate the site-packages directory for your my_env environment. This will output a path like C:\Users\my_user\anaconda3\envs\my_env\Lib\site-packages\:
```bash
conda activate my_env
python -c "import site; print(site.getsitepackages())"
```


    - Create a file named ecophys_utils.pth inside the folder found above, and add the following line to it:
	
```python
C:\Users\my_user\Documents\Github\ecophys_utils\
```


    - Now, you can simply use the library:

```python
from ecophysutils import dataloading
```

Would you like help organizing your functions into meaningful modules? 🚀

## How to Cite (Update later once useful)

Muller (2025). *ecophys_utils: Python utilities library for ecophysiology tasks*

## License

This software is distributed under the GNU GPL version 3

