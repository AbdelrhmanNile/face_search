import streamlit as st
import numpy as np
import PIL
import os
from sklearn.preprocessing import Normalizer

from modules.detection import MTCNNDetectionStrategy
from modules.representation import FaceNetEmbeddingStrategy
from modules.clustering import DBSCANClusteringStrategy

from modules.recognition import FaceRecognitionClient

IMAGES_DIR = "./src/face_search/faces"
OUTPUT_DIR = "./src/face_search/faces_clustered"
VECTOR_DB_PATH = "./src/face_search/faces_clustered/index.faiss"
FACENET_WEIGHTS = "./src/face_search/weights/facenet_keras_weights.h5"

client = FaceRecognitionClient(
    detection_strategy=MTCNNDetectionStrategy(),
    embedding_strategy=FaceNetEmbeddingStrategy(weights_path=FACENET_WEIGHTS),
    clustering_strategy=DBSCANClusteringStrategy(
        eps=0.5, min_samples=2, metric="cosine"
    ),
    l2_normalizer=Normalizer("l2"),
    images_dir=IMAGES_DIR,
    clusters_dir=OUTPUT_DIR,
    vector_db_path=VECTOR_DB_PATH,
    is_clustered=False,
    # please note that it will be slow if you set is_clustered to false
    # streamlit for some reason re runs the whole script several time and with is_clustered set to false
    # clustring will be done several times.
)


def main():
    # use webcam to get image
    st.title("Face Search")
    st.write(
        "This is a simple face search application. Use your webcam to retrieve similar faces."
    )

    # camera input
    captured = st.camera_input("Smile!!")

    if captured is not None:
        st.image(captured, caption="captured Image.", use_column_width=True)

        img = PIL.Image.open(captured)
        img = img.convert("RGB")
        img = np.array(img)

        if st.button("Search"):
            result = client.search(img, top_k=1)
            cluster_path = os.path.join(OUTPUT_DIR, str(result))
            st.write("Retrieved faces:")
            for image in os.listdir(cluster_path):
                img_path = os.path.join(cluster_path, image)
                st.image(img_path, use_column_width=True)


if __name__ == "__main__":
    main()
