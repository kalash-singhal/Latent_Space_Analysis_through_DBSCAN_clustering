import pandas as pd
import numpy as np
from LSTM import *
from connector import *
from Vanilla import *
from CNN import *
from Clustering_Analysis import *

def main():
    df_r = pd.DataFrame(np.random.randint(0,1000,size=(2676, 3322)), columns=range(3322))

    # CNN

    cnn = CNNAutoencoder(df_r)
    # change of input shape to (none, 2000)
    cnn_input_2d = cnn.resample()
    # scaling with zscore
    cnn_input_2d = CNNAutoencoder.z_score(cnn_input_2d)
    # conversion of 2d input to 3d input
    cnn_input_3d = CNNAutoencoder.into_3d(cnn_input_2d)
    # CNN Autoencoder with 3 layers and 64 units at bottleneck
    hist_cnn, cnn_model = cnn.cnn_autoencoder(cnn_input_3d)

    # creation of convolutional encoder between the input and bottleneck layer
    convo_encoder = CNNAutoencoder.cnn_encoder(cnn_model)
    # output of convolutional encoder
    cnn_encodings = Connector.predict(convo_encoder, cnn_input_3d)
    # output of convolutional autoencoder
    cnn_output = Connector.predict(cnn_model, cnn_input_3d)
    # conversion of 3d output into 2d output
    cnn_output = CNNAutoencoder.cnn_output_error(cnn_output)
    # reconstruction error between input and output of autoencoder
    cnn_recons_error = Connector.recons_error(cnn_output, cnn_input_2d.T)
    # mean reconstruction error from each time series signal
    cnn_mean_error = Connector.mean_error(cnn_recons_error)
    # print(cnn_mean_error)

    # CNN_DBSCAN Clustering analysis

    # Minmaxscaling of encodings (latent space) for DBSCAN
    cnn_encod = Connector.dbscan_scaling(cnn_encodings)
    cnn_dbscan = ClusterAnalysis(cnn_encod)
    # Calculation of mean euclidean distance of each signal and the cumulative anomalies for each signal
    lstm_eu_dist, lstm_anomalies = cnn_dbscan.run()

    # CNN-LSTM autoencoder

    # CNN encodings from above CNN Autoencoder is used
    cnn_lstm = LSTMAutoencoders(cnn_encodings)
    # scaling and sequence creation for stacked lstm autoencoder
    cnn_lstm_input = cnn_lstm.lstm_scaling()
    # stacked lstm Autoencoder with 2 layers and 2 units at bottleneck
    cnn_lstm_encoder, cnn_lstm_hist, cnn_lstm_model = cnn_lstm.stacked_lstmautoencoder(x_train=cnn_lstm_input)

    # output of cnn-lstm encoder
    cnn_lstm_encodings = Connector.predict(cnn_lstm_encoder, cnn_lstm_input)
    # output of cnn-lstm autoencoder
    cnn_lstm_output = Connector.predict(cnn_lstm_model, cnn_lstm_input)
    # output of the last sequence
    cnn_lstm_output = LSTMAutoencoders.lstm_output_error(cnn_lstm_output)
    # reconstruction error between input and output of autoencoder
    cnn_lstm_recons_error = Connector.recons_error(cnn_lstm_output, cnn_encodings)
    # mean reconstruction error from each time series signal
    cnn_lstm_mean_error = Connector.mean_error(cnn_lstm_recons_error)
    # print(cnn_lstm_mean_error)

    # CNN-LSTM_DBSCAN Clustering analysis

    # Minmaxscaling of encodings (latent space) for DBSCAN
    cnn_lstm_encod = Connector.dbscan_scaling(cnn_lstm_encodings)
    cnn_lstm_dbscan = ClusterAnalysis(cnn_lstm_encod)
    # Calculation of mean euclidean distance of each signal and the cumulative anomalies for each signal
    cnn_lstm_eu_dist, cnn_lstm_anomalies = cnn_lstm_dbscan.run()

    # Vanilla Autoencoder

    vanilla = VanillaAutoencoder(df_r)
    # scaling for vanilla autoencoder
    vanilla_input = vanilla.vanilla_scaling()
    # Vanilla Autoencoder with 3 layers and 64 units at bottleneck
    vanilla_encoder, vanilla_hist, vanilla_model = vanilla.vanilla_autoencoder(train_v=vanilla_input)

    # output of vanilla encoder
    vanilla_encodings = Connector.predict(vanilla_encoder, vanilla_input)
    # output of vanilla autoencoder
    vanilla_output = Connector.predict(vanilla_model, vanilla_input)
    # converted output
    vanilla_output = VanillaAutoencoder.vanilla_output_error(vanilla_output)
    # reconstruction error between input and output of autoencoder
    vanilla_recons_error = Connector.recons_error(vanilla_output, vanilla_input)
    # mean reconstruction error from each time series signal
    vanilla_mean_error = Connector.mean_error(vanilla_recons_error)
    # print(vanilla_mean_error)

    # Vanilla_DBSCAN Clustering analysis

    # Minmaxscaling of encodings (latent space) for DBSCAN
    vanilla_encod = Connector.dbscan_scaling(vanilla_encodings)
    vanilla_dbscan = ClusterAnalysis(vanilla_encod)
    # Calculation of mean euclidean distance of each signal and the cumulative anomalies for each signal
    vanilla_eu_dist, vanilla_anomalies = vanilla_dbscan.run()

    # Vanilla-lstm autoencoder

    # Vanilla encodings from above Vanilla Autoencoder is used
    vanilla_lstm = LSTMAutoencoders(vanilla_encodings)
    # scaling and sequence creation for stacked lstm autoencoder
    vanilla_lstm_input = vanilla_lstm.lstm_scaling()
    # stacked lstm Autoencoder with 2 layers and 2 units at bottleneck
    vanilla_lstm_encoder, vanilla_lstm_hist, vanilla_lstm_model = vanilla_lstm.stacked_lstmautoencoder(
        x_train=vanilla_lstm_input)

    # output of vanilla-lstm encoder
    vanilla_lstm_encodings = Connector.predict(vanilla_lstm_encoder, vanilla_lstm_input)
    # output of vanilla-lstm autoencoder
    vanilla_lstm_output = Connector.predict(vanilla_lstm_model, vanilla_lstm_input)
    # output of the last sequence
    vanilla_lstm_output = LSTMAutoencoders.lstm_output_error(vanilla_lstm_output)
    # reconstruction error between input and output of autoencoder
    vanilla_lstm_recons_error = Connector.recons_error(vanilla_lstm_output, vanilla_encodings)
    # mean reconstruction error from each time series signal
    vanilla_lstm_mean_error = Connector.mean_error(vanilla_lstm_recons_error)
    # print(vanilla_lstm_mean_error)

    # Vanilla-LSTM_DBSCAN Clustering analysis

    # Minmaxscaling of encodings (latent space) for DBSCAN
    vanilla_lstm_encod = Connector.dbscan_scaling(vanilla_lstm_encodings)
    vanilla_lstm_dbscan = ClusterAnalysis(vanilla_lstm_encod)
    # Calculation of mean euclidean distance of each signal and the cumulative anomalies for each signal
    vanilla_lstm_eu_dist, vanilla_lstm_anomalies = vanilla_lstm_dbscan.run()

    # LSTM Autoencoder

    # lstm input scaling before sequencing
    scaled_lstm = LSTMAutoencoders.lstm2_scaling(df_r)
    lstm = LSTMAutoencoders(scaled_lstm)
    # sequencing of lstm input
    lstm_input = lstm.lstm_scaling()
    # LSTM Autoencoder with 3 layers and 64 units at bottleneck
    lstm_hist, lstm_model, lstm_encoder = lstm.lstm_autoencoder(x_train=lstm_input)

    # output of lstm encoder
    lstm_encodings = Connector.predict(lstm_encoder, lstm_input)
    # output of lstm autoencoder
    lstm_output = Connector.predict(lstm_model, lstm_input)
    # output of the last sequence
    lstm_output = LSTMAutoencoders.lstm_output_error(lstm_output)
    # reconstruction error between input and output of autoencoder
    lstm_recons_error = Connector.recons_error(lstm_output, scaled_lstm)
    # mean reconstruction error from each time series signal
    lstm_mean_error = Connector.mean_error(lstm_recons_error)
    # print(lstm_mean_error)

    # LSTM_DBSCAN Clustering analysis

    # Minmaxscaling of encodings (latent space) for DBSCAN
    lstm_encod = Connector.dbscan_scaling(lstm_encodings)
    lstm_dbscan = ClusterAnalysis(lstm_encod)
    # Calculation of mean euclidean distance of each signal and the cumulative anomalies for each signal
    lstm_eu_dist, lstm_anomalies = lstm_dbscan.run()


if __name__ == "__main__":
    main()