import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import load_model, Model


class VanillaAutoencoder:

    def __init__(self, df):
        self.df = df
        self.batch_size = 64
        self.epochs = 20
        self.unit1 = 500
        self.unit2 = 200
        self.unit3 = 64

    # standardscaler
    @staticmethod
    def stdscale(dataf):
        return pd.DataFrame(StandardScaler().fit_transform(dataf))

    # Minmaxscaler
    @staticmethod
    def minmaxscale(dataf):
        return pd.DataFrame(MinMaxScaler().fit_transform(dataf))

    # Scaling of Vanilla input
    def vanilla_scaling(self):
        return self.minmaxscale(self.stdscale(self.df)).T

    # Vanilla Autoencoder with three layers and 64 as the bottleneck units
    def vanilla_autoencoder(self, train_v):
        input_abs1 = keras.Input(shape=(train_v.shape[1],))
        encoded = Dense(self.unit1, activation="tanh")(input_abs1)
        encoded = Dense(self.unit2, activation="tanh")(encoded)
        encoded = Dense(self.unit3, activation="tanh")(encoded)
        decoded = Dense(self.unit2, activation="tanh")(encoded)
        decoded = Dense(self.unit1, activation="tanh")(decoded)
        decoded = Dense(train_v.shape[1], activation="tanh")(decoded)

        autoencoder = keras.Model(input_abs1, decoded)
        encoder = keras.Model(input_abs1, encoded)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1,
                                                      mode='auto', baseline=None, restore_best_weights=True)

        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        autoencoder.summary()

        history = autoencoder.fit(train_v, train_v, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size,
                                  shuffle=False, verbose=1, callbacks=[early_stop])

        return encoder, history, autoencoder

    # To get suitable output for calculating the reconstruction error
    @staticmethod
    def vanilla_output_error(output):
        return pd.DataFrame(output)