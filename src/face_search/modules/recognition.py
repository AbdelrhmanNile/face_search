import cv2
from sklearn.preprocessing import Normalizer
import numpy as np
import os
import shutil
import faiss

from .utils import read_image, preprocess_face, normalize_embeddings
from .detection import FaceDetectionStrategy
from .representation import FaceEmbeddingStrategy
from .clustering import ClusteringStrategy


class FaceRecognitionClient:

    def __init__(
        self,
        detection_strategy: FaceDetectionStrategy,
        embedding_strategy: FaceEmbeddingStrategy,
        clustering_strategy: ClusteringStrategy,
        l2_normalizer: Normalizer,
        images_dir=None,
        clusters_dir=None,
        vector_db_path=None,
        is_clustered=False,
    ):

        # load strategies
        self.detection_strategy = detection_strategy
        self.embedding_strategy = embedding_strategy
        self.clustering_strategy = clustering_strategy
        self.l2_normalizer = l2_normalizer

        # if is_clustered is true, load the index
        if is_clustered:
            self.index = faiss.read_index(vector_db_path)
        else:  # if not re-cluster the images
            if os.path.exists(clusters_dir):
                shutil.rmtree(clusters_dir)
            self.cluster_faces(images_dir, clusters_dir)

    # given an image, detect and return the face
    def detect_face(self, img):
        return self.detection_strategy.detect_face(img)

    # given a face, return the embeddings
    def embed_face(self, face):
        return self.embedding_strategy.embed_face(face)

    # given an image, return the embeddings
    def get_face_embeddings(self, image):
        if isinstance(image, str):
            img = read_image(image)
        else:
            img = image
        face = self.detect_face(img)
        if face is None:
            return None
        face = preprocess_face(face)
        embeddings = self.embed_face(face)
        return normalize_embeddings(embeddings)

    # cluster the faces in the images_dir and save the clusters in the output_dir
    def cluster_faces(self, images_dir: str, output_dir: str, build_index=True):
        images = os.listdir(images_dir)
        embeddings = []

        # get embeddings for all the images
        for image in images:
            embeddings.append(self.get_face_embeddings(os.path.join(images_dir, image)))

        # convert to numpy array
        embeddings = np.concatenate(embeddings, axis=0)

        # cluster the embeddings, get the cluster labels
        labels = self.clustering_strategy.cluster(embeddings)

        # number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # for each cluster, create a folder
        for i in range(n_clusters):
            os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

        # move images to their respective clusters
        for i, label in enumerate(labels):
            if label == -1:
                continue
            image = images[i]
            shutil.copy(
                os.path.join(images_dir, image),
                os.path.join(output_dir, str(label), image),
            )

        # build faiss index
        if build_index:
            self.build_vector_index(output_dir)

    # build a faiss index, each cluster is represented by 1 embedding
    # only one image within a cluster is used to represent the cluster
    def build_vector_index(self, clustered_db_path):
        embeddings = []
        cluster_ids = os.listdir(clustered_db_path)
        cluster_ids = [
            int(cluster_id) for cluster_id in cluster_ids if cluster_id != "index.faiss"
        ]
        # sort the cluster ids to make sure the order is consistent
        cluster_ids.sort()

        # get 1 embedding from each cluster
        for cluster_id in cluster_ids:
            cluster_dir = os.path.join(clustered_db_path, str(cluster_id))
            images = os.listdir(cluster_dir)
            image = images[0]
            embedding = self.get_face_embeddings(os.path.join(cluster_dir, image))
            embeddings.append(embedding)

        # convert to numpy array
        embeddings = np.concatenate(embeddings, axis=0)

        # build L2 index, our embeddings are normalized using L2 "euclidean"
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # save the index
        faiss.write_index(index, os.path.join(clustered_db_path, "index.faiss"))
        self.index = index

    # match the query image with the clusters
    # return the cluster id
    def search(self, image_path, top_k=1):
        query_embedding = self.get_face_embeddings(image_path)
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0][0]
