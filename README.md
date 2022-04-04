# Latent_Space_Analysis_through_DBSCAN_clustering

This repository helps to compare the irregular changes in the clusters formed from the time-series data of force signals with the irregularities (wear) in the Fine-Blanking process.

Wear on the punch in the Fine-Blanking process causes a loss of money and time. To solve this problem: identification of the wear through time-dependent force sensor signals is carried out in this study. Identification of wear is done using latent spaces of five different autoencoders for comparative analysis. The five autoencoders are CNN, Vanilla, LSTM, CNN-LSTM, and Vanilla-LSTM.

The latent spaces form clusters in high-dimensional Euclidean space. Density-based DBSCAN clustering technique identifies clusters in the latent space based on the high-density points. Sparsely-populated points from the center of a cluster are considered anomalous points. Anomalousness in a cluster is the Euclidean distance of each point from a threshold (zero if a point is inside the threshold). The points inside the limit are considered more densely populated than the ones outside. The cumulative anomalousness of every point in the latent space is compared with the increase in the amount of wear during the fine-blanking operation.

These irregularities in clusters are compared with the wear data which is not uploaded due to privacy concerns.
