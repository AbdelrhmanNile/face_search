# Face Search

## Description
Quickly search for faces in a directory of images. <br />
 **this project uses:**
  1. MTCNN for face detection,
 2. FaceNet for face embeddings,
 3. DBscan for organizing the faces into clusters,
 4. fiass for searching the clusters for a specific face.

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