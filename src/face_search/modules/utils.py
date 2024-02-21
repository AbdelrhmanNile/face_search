import cv2
import numpy as np
from sklearn.preprocessing import Normalizer


def crop_face(img, x1, y1, x2, y2):
    return img[y1:y2, x1:x2]


def normalize_embeddings(embeddings):
    return Normalizer("l2").transform(embeddings)


def read_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_face(face):
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    return face
