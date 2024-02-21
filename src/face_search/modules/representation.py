from .facenet import InceptionResNetV2

import numpy as np
import cv2


class FaceEmbeddingStrategy:
    def embed_face(self, face):
        pass


class FaceNetEmbeddingStrategy(FaceEmbeddingStrategy):
    def __init__(
        self, weights_path="./src/face_search/weights/facenet_keras_weights.h5"
    ):
        self.embedder = InceptionResNetV2()
        self.embedder.load_weights(weights_path)

    def embed_face(self, face):
        face = face[0]
        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)
        return self.embedder.predict(face)
