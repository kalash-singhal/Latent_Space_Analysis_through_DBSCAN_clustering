import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, RepeatVector, Dropout, TimeDistributed, Reshape
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import load_model, Model, Sequential
from scipy.stats import zscore
from scipy import signal


class CNNAutoencoder:

    def __init__(self, df):
        self.df = df
        self.sample_rate = 2000
        self.batch_size = 64
        self.epochs = 20
        self.filter1 = 1000
        self.filter2 = 100
        self.filter3 = 4
        self.kernel_size = 7
        self.last_filter = 1
        self.strides = 5
        self.bottleneck = 64

    # downsamples a given timeseries to the number of datapoints given in sample_rate (2000)
    def resample(self):
        if self.sample_rate == 0:
            return self.df
        else:
            df_temp = pd.DataFrame()
            df_comb = pd.DataFrame()
            count = 0
            for segment in self.df:
                down_sampled = signal.resample(self.df[segment], self.sample_rate)
                df_temp = pd.DataFrame({count: down_sampled})
                df_comb = pd.concat([df_comb, df_temp], axis=1)
                count = count + 1
            return df_comb

    # applies the zcore to each cell of dataframe
    @staticmethod
    def z_score(df):
        return df.apply(zscore)

    # convert input from 2d to 3d
    @staticmethod
    def into_3d(data):
        input_r = data.T.values
        input_r.shape = input_r.shape + (1,)
        return input_r

    # the filter size, kernel_size, and strides of each layer is calculated based on the input size of (none, 2000, 1)
    # the output size from the bottleneck layer needs to be 64 for comparative analysis with other autoencoders
    # Also a dense layer is added after the bottleneck layer to get the features of latent
    # space without any spatial influence
    def cnn_autoencoder(self, input):
        model = keras.Sequential(
            [
                layers.Input(shape=(input.shape[1], input.shape[2])),  # definition of the input layer
                layers.Conv1D(
                    filters=self.filter1, kernel_size=self.kernel_size, padding="same", strides=self.strides, activation="tanh"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1D(
                    filters=self.filter2, kernel_size=self.kernel_size, padding="same", strides=self.strides, activation="tanh"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1D(
                    filters=self.filter3, kernel_size=self.kernel_size, padding="same", strides=self.strides, activation="tanh"
                ),
                layers.Flatten(),
                layers.Dense(self.bottleneck, activation="tanh", name="Flat"),
                # bottleneck of the autoencoder, name is set to "Flat"
                layers.Reshape((int(input.shape[1] / 125), self.filter3)),
                layers.Conv1DTranspose(
                    filters=self.filter3, kernel_size=self.kernel_size, padding="same", strides=self.strides, activation="tanh"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=self.filter2, kernel_size=self.kernel_size, padding="same", strides=self.strides, activation="tanh"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=self.filter1, kernel_size=self.kernel_size, padding="same", strides=self.strides, activation="tanh"
                ),
                layers.Conv1DTranspose(filters=self.last_filter, kernel_size=self.kernel_size, padding="same"),
            ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        # visualisierung keras
        # keras.utils.model_to_dot(model)
        model.summary()

        history = model.fit(input, input, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, shuffle=False,
                            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min",
                                                                     restore_best_weights=True)])

        return history, model

    # CNN encoder from the "Flat" layer of the CNN autoencoder
    @staticmethod
    def cnn_encoder(model_cnn):
        # compute the output of flattened enc layer (bottleneck)
        flat_layer = model_cnn.get_layer('Flat')
        encoder = keras.Model(model_cnn.input, flat_layer.output)
        return encoder

    # converting the output from 3d to 2d
    @staticmethod
    def cnn_output_error(output):
        return pd.DataFrame(output[:, :, 0])  # changed into 2d

