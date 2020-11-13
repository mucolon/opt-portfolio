import numpy as np
import pandas as pd
import os
import json
# from alpha_vantage.timeseries import TimeSeries
# ts = TimeSeries(key='RFNKYQY0BFFR03I9')


# get current working directory
cwd = os.getcwd()

path_test = cwd + "/test/test.json"

with open(path_test) as f:
    data = json.load(f)


for key in data.keys():
    print(key)
