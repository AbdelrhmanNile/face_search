from face_recognition import FaceRecognitionClient
import streamlit as st
import os
import PIL
import numpy as np


images_dir = "./src/face_search/faces"
output_dir = "./src/face_search/faces_clustered"
vector_db_path = "./src/face_search/faces_clustered/index.faiss"

fr = FaceRecognitionClient(
    vector_db_path=vector_db_path,
    images_dir=images_dir,
    clusters_dir=output_dir,
    is_clustered=True,  # set to false if you want to re cluster the images
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
            result = fr.search(img, top_k=1)
            cluster_path = os.path.join(output_dir, str(result))
            print(cluster_path)
            print("#############")
            st.write("Retrieved faces:")
            for image in os.listdir(cluster_path):
                img_path = os.path.join(cluster_path, image)
                st.image(img_path, use_column_width=True)


if __name__ == "__main__":
    main()
