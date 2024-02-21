import mtcnn
import keras
from arch import InceptionResNetV2
import cv2
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
import numpy as np
import os
import shutil
import faiss


class FaceRecognitionClient:
    def __init__(
        self,
        vector_db_path=None,
        images_dir=None,
        clusters_dir=None,
        is_clustered=False,
    ):
        # load face detection model
        self.detector = mtcnn.MTCNN()

        # load facenet model for face embeddings
        self.embedder = InceptionResNetV2()
        self.embedder.load_weights("./src/face_search/facenet_keras_weights.h5")

        # l2 normalizer for face embeddings
        self.l2_normalizer = Normalizer("l2")

        if is_clustered:
            self.index = faiss.read_index(vector_db_path)
        else:
            # remove existing clusters
            if os.path.exists(clusters_dir):
                shutil.rmtree(clusters_dir)
            self.cluster_faces(images_dir, clusters_dir)

    def crop_face(self, img, x1, y1, x2, y2):
        return img[y1:y2, x1:x2]

    # normalize embeddings
    def normalize_embeddings(self, embeddings):
        return self.l2_normalizer.transform(embeddings)

    def read_image(self, image_path: str):
        # read image
        img = cv2.imread(image_path)
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess_face(self, face):
        # resize image
        face = cv2.resize(face, (160, 160))
        # normalize
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        # add batch dimension
        face = np.expand_dims(face, axis=0)
        return face

    # detecs face in image and return cropped face
    def detect_face(self, img):
        # detect
        detected_faces = self.detector.detect_faces(img)
        # if no face is detected return None
        if len(detected_faces) == 0:
            return None
        # crop face
        x1, y1, width, height = detected_faces[0]["box"]
        x2, y2 = x1 + width, y1 + height
        return self.crop_face(img, x1, y1, x2, y2)

    def embed_face(self, face):
        return self.embedder.predict(face)

    def get_face_embeddings(self, image):
        if isinstance(image, str):
            img = self.read_image(image)
        else:
            img = image
        face = self.detect_face(img)
        if face is None:
            return None
        face = self.preprocess_face(face)
        embeddings = self.embed_face(face)
        return self.normalize_embeddings(embeddings)

    # given a directory of images, cluster same faces into a folder together
    def cluster_faces(self, images_dir: str, output_dir: str, build_index=True):

        images = os.listdir(images_dir)
        embeddings = []

        # read images and extract faces
        for image in images:
            embeddings.append(self.get_face_embeddings(os.path.join(images_dir, image)))

        # convert to numpy array
        embeddings = np.concatenate(embeddings, axis=0)

        # cluster faces using DBSCAN, because we don't know the number of clusters
        clustering_model = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
        clustering_model.fit(embeddings)
        # get cluster labels
        labels = clustering_model.labels_
        # number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # create a folder for each cluster
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

        # build index
        if build_index:
            self.build_vector_index(output_dir)

    def build_vector_index(self, clustered_db_path):
        # from each cluster, get 1 embedding
        embeddings = []
        cluster_ids = os.listdir(clustered_db_path)
        cluster_ids = [
            int(cluster_id) for cluster_id in cluster_ids if cluster_id != "index.faiss"
        ]
        cluster_ids.sort()
        for cluster_id in cluster_ids:
            cluster_dir = os.path.join(clustered_db_path, str(cluster_id))
            images = os.listdir(cluster_dir)
            image = images[0]
            embedding = self.get_face_embeddings(os.path.join(cluster_dir, image))
            embeddings.append(embedding)

        # convert to numpy array
        embeddings = np.concatenate(embeddings, axis=0)

        # build index, use euclidean distance
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # save index
        faiss.write_index(index, os.path.join(clustered_db_path, "index.faiss"))
        self.index = index

    def search(self, image_path, top_k=1):
        # get face embeddings
        query_embedding = self.get_face_embeddings(image_path)

        # search
        distances, indices = self.index.search(query_embedding, top_k)

        return indices[0][0]
