# Face Search

## Description
Quickly search for faces in a directory of images. <br />
 **this project uses:**
  1. MTCNN for face detection,
 2. FaceNet for face embeddings,
 3. DBscan for organizing the faces into clusters,
 4. faiss for searching the clusters for a specific face.

## Installation
to reproduce the development environment, use PDM
```bash
pip install pdm
```
then `cd` into the project directory and run
```bash
pdm install
```

## Usage
`cd` into the project directory and run
```bash
pdm run streamlit run src/face_search/app.py
```
ِi provided 2 examples to test on `q_1.jpg` and `q_2.jpg` in the `src/face_search` directory.


https://github.com/AbdelrhmanNile/face_search/assets/90456995/c227446e-1d94-4eb9-8b8e-5af56589b409


### Using web cam

to test it with webcam and your own pictures, do the following:
1. place your pictures in the `src/face_search/faces` directory, minimum 2 pictures.
2. run the web cam version of the app by running
```bash
pdm run streamlit run src/face_search/app_webcam.py
```
**Note:** the webcam version will be SLOWER, streamlit issue.
