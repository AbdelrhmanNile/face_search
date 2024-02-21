# webcam version of the app
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from face_recognition import FaceRecognitionClient
import os
import PIL


images_dir = "./src/face_search/faces"
output_dir = "./src/face_search/faces_clustered"
vector_db_path = "./src/face_search/faces_clustered/index.faiss"

fr = FaceRecognitionClient(
    vector_db_path=vector_db_path,
    images_dir=images_dir,
    clusters_dir=output_dir,
    is_clustered=False,
    # please note that it will be slow if you set is_clustered to false
    # streamlit for some reason re runs the whole script several time and with is_clustered set to false
    # clustring will be done several times.
)

print("Face Recognition Client initialized")


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
            result = fr.search(img, top_k=1)
            cluster_path = os.path.join(output_dir, str(result))
            st.write("Retrieved faces:")
            for image in os.listdir(cluster_path):
                img_path = os.path.join(cluster_path, image)
                st.image(img_path, use_column_width=True)


main()
