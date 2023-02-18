#
# Copyright (C) Remittan (remittan.com), 2023.
# All rights reserved.
# Ehsan Haghpanah; haghpanah@remittan.com
#

# %%

import pandas as pa
import os
from IPython.display import display

try:
     path: str = os.getcwd() + '\\data\\bitcoin-dataset.xlsx'
     df = pa.read_excel(path)
     df['Daily Return'] = df['Bitcoin Market Price USD'].pct_change()
     # display(df)
     
     print(df['Daily Return'].max())
     print(df['Daily Return'].min())
          
except Exception as p:
     print(f'error = {p.args}')
     pass

# %%
