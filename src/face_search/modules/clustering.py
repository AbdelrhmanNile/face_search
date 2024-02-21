from sklearn.cluster import DBSCAN


class ClusteringStrategy:
    def cluster(self, embeddings):
        pass


class DBSCANClusteringStrategy(ClusteringStrategy):
    def __init__(self, eps, min_samples, metric):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def cluster(self, embeddings):
        clustering_model = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric=self.metric
        )
        clustering_model.fit(embeddings)
        return clustering_model.labels_
