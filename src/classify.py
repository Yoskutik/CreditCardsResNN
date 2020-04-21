from tensorflow.keras import models
from getopt import getopt
import pandas as pd
import numpy as np
import pickle
import sys


def get_opts():
    opts = {}
    options, args = getopt(sys.argv[1:], 'po:f:')
    for key, value in options:
        opts[key.replace('-', '')] = value
    return opts, args


opts, args = get_opts()

model = models.load_model('model.h5')

with open('scaler.pickle', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

df = pd.read_csv(opts['f'])
df[['Amount']] = sc.transform(df[['Amount']])

y_pred = model.predict(df.values)

if 'p' not in opts.keys():
    y_pred = np.where(y_pred > 0.5, 1, 0)

if 'o' in opts.keys():
    with open(opts['o'], 'w') as file:
        file.write(str(y_pred))
else:
    print(y_pred)