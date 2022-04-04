import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from kneed import KneeLocator


class ClusterAnalysis:

    def __init__(self, input):
        self.input = input
        self.min_samples = 18
        self.mean_shape = 64
        self.intervals = 200
        self.lower_limit = 0.8
        self.permissible_limit = 0.9

    # calculate distances from NeareestNeighbors to get the knee value for DBSCAN clustering
    def knee_locator(self):
        neigh = NearestNeighbors(n_neighbors=2)
        near_n = neigh.fit(self.input)
        distances, indices = near_n.kneighbors(self.input)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        kn = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        return distances, distances[kn.knee]

    # Label the clusters by implementation of DBSCAN from the knee value
    def dbscan_clustering(self, knee):
        dbs = DBSCAN(eps=knee, min_samples=self.min_samples)
        labels = dbs.fit_predict(self.input)
        return labels

    # conversion of input into labelled dataframe
    def labelled_dataframe(self, labels):
        df = pd.DataFrame(self.input)
        df['Cluster'] = labels
        return df

    # calculate the total number of clusters including the "-1" cluster
    @staticmethod
    def total_clusters(labels):
        return len(np.unique(labels))

    # Find the centres of all clusters except the "-1" cluster
    def find_centers(self, df, cluster):
        centers = {}
        for x in np.arange(0, (cluster - 1), 1):
            centers[x] = df[df['Cluster'] == x].iloc[:, :self.mean_shape].mean(axis=0)
        return pd.DataFrame(centers).T

    # Reallocate the labels of "-1" cluster to get the sparsely populated points for each clsuter
    # methodology of minimum euclidean distance is used for reallocation
    def reallocation(self, df, cluster):
        centers = self.find_centers(df, cluster)
        if cluster == 2:
            df.loc[(df['Cluster'] == -1), 'Cluster'] = 0
        else:
            for index, row in df[df['Cluster'] == -1].iterrows():
                dic = {}
                for i in np.arange(0, (cluster - 1), 1):
                    dic[i] = euclidean(centers.loc[i], row.iloc[:self.mean_shape])
                df.loc[index, 'Cluster'] = min(dic, key=dic.get)

    # After reallocation, it calculates the euclidean distance of each point from its respective cluster centre
    def eu_dist(self, df, cluster):
        dist = []
        centers = self.find_centers(df, cluster)
        for index, row in df.iterrows():
            dist.append(euclidean(centers.loc[row.loc['Cluster']], row.iloc[:self.mean_shape]))
        df['dist'] = dist

    # Calculates the mean euclidean distance
    @staticmethod
    def mean_eu_dist(df):
        return np.mean(df['dist'])

    # Gives a list of percentage of points and the increasing rate of distance
    # (increasing radius) from the centre of a cluster
    def limit_graphs(self, df, length, cluster):
        dist = np.linspace(0, length, num=self.intervals)
        percentage = []
        for n in dist:
            count = 0
            for row in df[df['Cluster'] == cluster].index:
                if df['dist'][row] < n:
                    count += 1
                else:
                    continue
            percentage.append(count / len(df[df['Cluster'] == cluster]))

        # assert isinstance(dist, object)
        return percentage, list(dist)

    # set the limits of each cluster by separating densely and sparsely populated regions
    # It gives the threshold distance from the center for calculating anomalous distance
    # of each sparsely located points
    def right_limits(self, df, clusters):
        dicts = {}
        values = []
        high_dist = round(np.max(df['dist']), 2)
        for clu in range(0, (clusters - 1)):
            percentage, dist = self.limit_graphs(df, length=high_dist, cluster=clu)
            per = [x for i, x in enumerate(percentage) if x > self.lower_limit]
            new_dist = [dist[x] for x in [i for i, x in enumerate(percentage) if x > self.lower_limit]]
            kn = KneeLocator(new_dist, per, curve='concave', direction='increasing')
            if kn.knee is None:
                values.append(min([dist[x] for x in [i for i, x in enumerate(percentage) if x >
                                                     self.permissible_limit]]))
            else:
                values.append(kn.knee)

        for clu in range(0, (clusters - 1)):
            dicts[clu] = values[clu]

        return dicts

    # Calculates the anomalous distance of each point outside the threshold
    @staticmethod
    def anomalies_dist(clusters, dic, df):
        df['ano_dis'] = 0.0
        for clu in range(0, (clusters - 1)):
            df.loc[(df['Cluster'] == clu) & (df['dist'] >= dic[clu]), 'ano_dis'] = \
                df.loc[(df['Cluster'] == clu) & (df['dist'] >= dic[clu])]['dist'] - dic[clu]

    # cumulative addition of anomalous distance for each point
    @staticmethod
    def cum_anomalies(df):
        return df['ano_dis'].cumsum(axis=0)

    # run method calulates the mean euclidean distance and the cumulative anomalies
    # for the input encodings (latent space)
    def run(self):
        eps_distance, knee = self.knee_locator()
        labels = self.dbscan_clustering(knee)
        df = self.labelled_dataframe(labels)
        clusters = self.total_clusters(labels)
        self.reallocation(df, clusters)
        self.eu_dist(df, clusters)
        mean_eu_dist = self.mean_eu_dist(df)
        # print(mean_eu_dist)

        limits = self.right_limits(df, clusters)
        self.anomalies_dist(clusters, limits, df)
        anomalies = self.cum_anomalies(df)
        return mean_eu_dist, anomalies