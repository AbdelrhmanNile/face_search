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
    is_clustered=True,  # set to False if you want to re-cluster the images
    # please note that it will be slow if you set is_clustered to false
    # streamlit for some reason re runs the whole script several time and with is_clustered set to false
    # clustring will be done several times.
)


def main():
    st.title("Face Search")

    st.write(
        "This is a simple face search application. Upload an image and retrieve similar faces."
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        img = PIL.Image.open(uploaded_file)
        img = img.convert("RGB")
        img = np.array(img)

        if st.button("Search"):
            result = client.search(img, top_k=1)
            cluster_path = os.path.join(OUTPUT_DIR, str(result))

            st.write("Retrieved images:")
            for image in os.listdir(cluster_path):
                img_path = os.path.join(cluster_path, image)
                st.image(img_path, use_column_width=True)


if __name__ == "__main__":
    main()
