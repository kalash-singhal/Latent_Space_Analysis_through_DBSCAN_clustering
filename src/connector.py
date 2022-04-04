import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class Connector(object):

    # predict output of input given to encoder/autoencoder
    @staticmethod
    def predict(encoder, train):
        return encoder.predict(train)

    # Reconstruction error between input and output signal
    @staticmethod
    def recons_error(output, input):
        input = pd.DataFrame(input)
        mse = []
        for row in zip(output.index, input.index):
            mse.append(mean_squared_error(output.loc[row, :], input.loc[row, :]))
        return mse

    # Mean reconstruction error
    @staticmethod
    def mean_error(error_lstm):
        return np.mean(error_lstm)

    # Minmaxscaling for DBSCAN input
    @staticmethod
    def dbscan_scaling(encodings):
        return MinMaxScaler().fit_transform(encodings)


