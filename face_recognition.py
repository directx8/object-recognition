import mediapipe as mp
import cv2
import numpy as np


class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3, min_detection_confidence=0.3, min_tracking_confidence=0.3) # up to 3 faces

    def blur_facial_features(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_face_mesh = self.face_mesh.process(frame_rgb)

        if result_face_mesh.multi_face_landmarks:
            for id, face_landmarks in enumerate(result_face_mesh.multi_face_landmarks):
                print(id)
                cx_min = width
                cy_min = height
                cx_max = cy_max = 0
                for id, lm in enumerate(face_landmarks.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
                try:
                    frame[cy_min:cy_max, cx_min:cx_max] = cv2.blur(frame[cy_min:cy_max, cx_min:cx_max], (27,27), cv2.INTER_AREA)
                except Exception as e:
                    print(e)
                    pass
        return frame
