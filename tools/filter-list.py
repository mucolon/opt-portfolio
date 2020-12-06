# file: remove-duplicates.py
# purpose: remove duplicates from watchlist

import os
import numpy as np
import pandas as pd

# file paths
cwd = os.getcwd()
path_list = cwd + "/tools/watchlist.txt"
path_ignore = cwd + "/data/ignore-simple.csv"

# remove duplicates from ignore column
df = pd.read_csv(path_ignore)
ignore_ = df["Ignore"].tolist()
init_len = len(ignore_)
_ = []
[_.append(x) for x in ignore_ if x not in _]
ignore_ = _
fin_len = len(ignore_)
diff_len = -1*(fin_len - init_len)
df["Ignore"] = ignore_ + [np.nan]*diff_len
df.dropna(how="all", inplace=True)
df.to_csv(path_ignore, index=False)

# remove duplicates from watchlist
# remove ignore-simple.csv symbols from watchlist
ignore_ = [str.lower() for str in ignore_]
list_ = ignore_[:]
with open(path_list, "r+") as f:
    raw_list = [str.split(",") for str in f][0]
    [list_.append(x) for x in raw_list if x not in list_]
    list_ = [x for x in list_ if x not in set(ignore_)]
    f.seek(0)
    f.write(','.join(list_))
    f.truncate()
