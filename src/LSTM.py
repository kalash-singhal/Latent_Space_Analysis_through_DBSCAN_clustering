import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, TimeDistributed, Reshape
from tensorflow.keras import layers
from tensorflow.python.keras.layers import LSTM
from tensorflow.keras.models import load_model, Model, Sequential


class LSTMAutoencoders:

    def __init__(self, encoding):
        self.encoding = encoding
        self.batch_size = 32
        self.epochs = 20
        self.unit1 = 16
        self.unit2 = 2
        self.unit3 = 1000
        self.unit4 = 500
        self.unit5 = 64
        self.seq_len = 12

    # sequencing for LSTM input
    @staticmethod
    def series_to_supervised(data, features, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        for i in range(n_in):
            data = np.insert(data, [0] * features, 0)
        data = data.reshape(int(data.shape[0] / features), features)
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, -1, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # Drop unused columns
    @staticmethod
    def dropp(seq_len, features, data):
        drop = [i for i in range(seq_len * features + 1, ((seq_len + 1) * features))]
        data.drop(data.columns[drop], axis=1, inplace=True)
        return data.values

    # reshape the sequences in the required shape
    @staticmethod
    def reshape(data, seq_len, features):
        train = data[:, 0:-1]
        train = train.reshape(train.shape[0], seq_len, features)
        return train

    # complete sequencing combining above methods
    def lstm_scaling(self):
        reframed = self.series_to_supervised(self.encoding, features=self.encoding.shape[1], n_in=(self.seq_len-1), n_out=1)
        train = self.dropp(seq_len=self.seq_len, features=self.encoding.shape[1], data=reframed)
        train_set = self.reshape(data=train, seq_len=self.seq_len, features=self.encoding.shape[1])
        return train_set

    # stacked LSTM Autoencoder with 2 layers and bottleneck of 2 units
    def stacked_lstmautoencoder(self, x_train):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1,
                                                      mode='auto', baseline=None, restore_best_weights=True)

        inputs = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
        encoded = LSTM(self.unit1, batch_input_shape=(None, x_train.shape[1], x_train.shape[2]), return_sequences=True)(
            inputs)
        encoded = Dropout(rate=0.2)(encoded)
        encoded1 = LSTM(self.unit2, return_sequences=False)(encoded)
        decoded = RepeatVector(x_train.shape[1])(encoded1)
        decoded = LSTM(self.unit2, return_sequences=True)(decoded)
        decoded = Dropout(rate=0.2)(decoded)
        decoded = LSTM(self.unit1, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(x_train.shape[2]))(decoded)

        auto = Model(inputs, decoded)
        encoder = Model(inputs, encoded1)

        auto.compile(loss='mean_squared_error', optimizer='Adam')
        encoder.compile(loss='mean_squared_error', optimizer='Adam')

        auto.summary()
        encoder.summary()

        history = auto.fit(x_train, x_train, validation_data=(x_train, x_train), epochs=self.epochs, batch_size=self.batch_size,
                           shuffle=False, verbose=1, callbacks=[early_stop])

        return encoder, history, auto

    # LSTM Autoencoder with 3 layers and bottleckneck of 64 units
    def lstm_autoencoder(self, x_train):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1,
                                                      mode='auto', baseline=None, restore_best_weights=True)

        inputs = layers.Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))
        encoded = layers.LSTM(self.unit3, batch_input_shape=(None, x_train.shape[1], x_train.shape[2]),
                              return_sequences=True)(inputs)
        encoded = layers.Dropout(rate=0.2)(encoded)
        encoded = layers.LSTM(self.unit4, return_sequences=True)(encoded)
        encoded = layers.Dropout(rate=0.2)(encoded)
        encoded1 = layers.LSTM(self.unit5, return_sequences=False, name="Flat")(encoded)
        decoded = layers.RepeatVector(x_train.shape[1])(encoded1)
        decoded = layers.LSTM(self.unit5, return_sequences=True)(decoded)
        decoded = layers.LSTM(self.unit4, return_sequences=True)(decoded)
        decoded = layers.LSTM(self.unit3, return_sequences=True)(decoded)
        decoded = layers.TimeDistributed(Dense(x_train.shape[2]))(decoded)
        # decoded = layers.LSTM(input_dim,stateful = False,return_sequences = True)(decoded)

        model = Model(inputs, decoded)
        encoder = Model(inputs, encoded1)

        model.compile(loss='mean_squared_error', optimizer='Adam')
        encoder.compile(loss='mean_squared_error', optimizer='Adam')

        model.summary()
        encoder.summary()

        history = model.fit(x_train, x_train, validation_data=(x_train, x_train), epochs=self.epochs, batch_size=self.batch_size,
                            shuffle=False, verbose=1, callbacks=[early_stop])

        return history, model, encoder

    # get the output of the last sequence
    @staticmethod
    def lstm_output_error(output):
        return pd.DataFrame(output[:, -1, :])  # last sequence contain each data point

    # initial scaling for LSTM Autoencoder
    @staticmethod
    def lstm2_scaling(df):
        return StandardScaler().fit_transform(df).T