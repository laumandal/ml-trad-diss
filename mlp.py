
#%% imports 
import technicals as t
import cloudfiles as c
import labelling as l

import pandas as pd
import numpy as np
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-colorblind')
plt.style.use('seaborn-whitegrid')

from sklearn.metrics import confusion_matrix

#%% get file from gcloud
creds = pathlib.Path("/Users/laumandal/downloads/Real-Time Data Analysis-53764f4812aa.json")
bucket = 'hist_datasets'
filepath = pathlib.Path('eod/2019H1 30y/EURUSD_EOD.csv')
c.dl_from_gcs(creds,bucket,filepath,localfolder='downloads')

#%% feature extraction

dlfile = pathlib.Path('downloads') / pathlib.Path(filepath.name)
df = pd.read_csv(dlfile,header=1, index_col=0)
#drop values with N/A in index column (some left in csv from diff size data in excel)
df = df[df.index.notnull()]
df.index = pd.to_datetime(df.index, format="%d/%m/%Y") # V IMPORTANT
df = df[['PX_LAST']]

#%% Add technicals to the dataframe

# [LAU NOTE: if want to train on multiple instruments' time series, need to normalize]
# [i.e. moving average etc must be on returns]

px = 'PX_LAST'

df['ret130'] = t.ret(df[px],130)
df['ret261'] = t.ret(df[px],261)
df['ma200'] = t.ma(df[px],200)
df['ma100'] = t.ma(df[px],100)
df['ma50'] = t.ma(df[px],50)
df['xover5_200'] = t.xover(df[px],5,200)
df['xover50_200'] = t.xover(df[px],50,200)
df['xover5_100'] = t.xover(df[px],5,100)
df['xover10_200'] = t.xover(df[px],10,200)
df['up2d261'] = t.up2down(df[px],261)
df['up2d130'] = t.up2down(df[px],130)
df['up2d65'] = t.up2down(df[px],65)
df['macd'] = t.macd(df[px])
df['rsi14'] = t.rsi(df[px],14)
df['rsi20'] = t.rsi(df[px],20)
df['rsi50'] = t.rsi(df[px],50)

#%% visualization
fig, ax = plt.subplots(1,1,figsize=(18,6))
df[['PX_LAST', 'ma50']].plot(ax=ax)
plt.show()

#%% labelling
upper=0.03
lower=0.03
timeout = pd.to_timedelta(2, unit='W')

out = l.triple_barrier_label(df['PX_LAST'],upper=upper, lower=lower, timeout = timeout)

#%% append labels to original df, drop NAs
df['label']=out['label']
df = df.dropna()

#%% visualize labels
fig, ax = plt.subplots(1,1,figsize=(18,6))
ax.scatter(out.index, out['label'], alpha=0.2)
#plt.show())

#%% build model

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)

train, test = train_test_split(df, test_size=0.2, shuffle=False)
train, val = train_test_split(train, test_size=0.2, shuffle=False)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(dataframe, labelcol, shuffle=True, batch_size=32, ):
    dataframe = dataframe.copy()
    labels = dataframe.pop(labelcol)
    #convert labels from [-1,0,1] to [1,0,0],[0,1,0],[0,0,1]
    labels = tf.keras.utils.to_categorical(labels+1)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

#Take all columns as numeric except last one (label)
numeric_cols = list(train.columns)[0:-1]

feature_columns = []
for header in numeric_cols:
    feature_columns.append(tf.feature_column.numeric_column(header))

# Now that we have defined our feature columns, 
# we will use a DenseFeatures layer to input them to our Keras model.
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train,'label', batch_size=batch_size)
val_ds = df_to_dataset(val,'label', shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test,'label', shuffle=False, batch_size=batch_size)

# build network
tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

#%%  Check accuracy
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

#convert classes back to [-1,0,1]
predictions = model.predict(test_ds)
classes = np.argmax(predictions, axis=1) - 1

#Put predictions back on the test set pandas dataframe
test['predictions'] = classes

#Check confusion matrix
c=confusion_matrix(y_true=test.label, y_pred=test.predictions, labels=[-1,0,1])
print(c)



###
# TO EVALUATE:
# Precision
# Recall
# F1

# config file to specify tests:
# - contains a list of input files
# - contains a list of models (or locations of files)
# - contains a list of evaluations wanted


#%%
