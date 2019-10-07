
# First we try a univariate time series; predict prices just based on historical.
# Then we will try a multivariate: technical indicators AND prices

# %%
# from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import pathlib
import technicals as t

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# %%
# for now reuse csv downloaded in mlp file

csv_path = pathlib.Path('downloads/EURUSD_EOD.csv')
df = pd.read_csv(csv_path, header=1, index_col=0)
# drop values with N/A in index column (some left in csv from diff size data in excel)
df = df[df.index.notnull()]
df.index = pd.to_datetime(df.index, format="%d/%m/%Y")  # V IMPORTANT


# %% PART 1: UNIVARIATE

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    """prepares examples to train on"""

    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


# %%

TRAIN_SPLIT = 6000  # need to merge datasets and have many more
tf.random.set_seed(13)  # for reproducability

uni_data = df['PX_LAST'].values

# normalize data (using only train part of course)
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

# normalize by subtracting mean and dividing by sd
# LAU NOTE: Does this make sense for prices? maybe better to just convert to returns
uni_data = (uni_data-uni_train_mean)/uni_train_std

# %% create test and train data to train the model

# the model is given the last 20 points and asked to predict the next one
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

# %% shuffle, batch, and cache the dataset
# LAU NOTE: look into the functions used in this part

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices(
    (x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


# %% compile the model

# starts off with a simple 1 layer model

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

# %%
# [LAU NOTE: train on full set when doing properly!]
EVALUATION_INTERVAL = 200  # only use to save time for examples
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)

# %% vizualization functions


def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(
            ), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


# %%
for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(),
                      simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()

# %% PART 2: MULTIVARIATE:

# first we add on the technicals to the df so they can be used
px = 'PX_LAST'

df['ret130'] = t.ret(df[px], 130)
df['ret261'] = t.ret(df[px], 261)
df['ma200'] = t.ma(df[px], 200)
df['ma100'] = t.ma(df[px], 100)
df['ma50'] = t.ma(df[px], 50)
df['xover5_200'] = t.xover(df[px], 5, 200)
df['xover50_200'] = t.xover(df[px], 50, 200)
df['xover5_100'] = t.xover(df[px], 5, 100)
df['xover10_200'] = t.xover(df[px], 10, 200)
df['up2d261'] = t.up2down(df[px], 261)
df['up2d130'] = t.up2down(df[px], 130)
df['up2d65'] = t.up2down(df[px], 65)
df['macd'] = t.macd(df[px])
df['rsi14'] = t.rsi(df[px], 14)
df['rsi20'] = t.rsi(df[px], 20)
df['rsi50'] = t.rsi(df[px], 50)


# %% choose features to include in the multivariate prediction:
features_considered = ['PX_LAST', 'ret130', 'macd']

features = df[features_considered].dropna()
features.head()


# %% normalize
# [LAU NOTE: really necceasry for these things? ...]
dataset = features.values
data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)
dataset = (dataset-data_mean)/data_std

# %% SINGLE STEP MODEL
# Do a 1 step prediction

#similar to univariate, but it samples based on the step size as well
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


#%% prepare data
# shown past data (23 bd) for previous month
# make a prediction for 11bd in future

past_history = 23
future_target = 11
STEP = 1

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)


#%% create datasets in tensorflow format
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


#%% create model

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

#%% train and store training history

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)


#%% function to visualize training history

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


#%%
plot_train_history(single_step_history,
                   'Single Step Training and validation loss')

#%% predict a single step future

for x, y in val_data_single.take(3):
  plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Single Step Prediction')
  plot.show()


#%% MULTI STEP MODEL
# predict daily price for the following 11 days

#as before, create data sets and then put in tensorflow format

future_target = 11
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

#%% plot a sample data point

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()


#%%
for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))

#%% define the model: 2 LSTM layers now, and dense layer outputs however many
# predictions you are trying to make (eg daily 11bd predicions is 11)

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(11))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')


#%% train model

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)


#%%
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

#%% look at how good the predictions are:
for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


#%%
