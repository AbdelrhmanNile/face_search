import mtcnn


class FaceDetectionStrategy:
    def detect_face(self, img):
        pass


class MTCNNDetectionStrategy(FaceDetectionStrategy):
    def __init__(self):
        self.detector = mtcnn.MTCNN()

    def detect_face(self, img):
        detected_faces = self.detector.detect_faces(img)
        if len(detected_faces) == 0:
            return None
        x1, y1, width, height = detected_faces[0]["box"]
        x2, y2 = x1 + width, y1 + height
        return img[y1:y2, x1:x2]
