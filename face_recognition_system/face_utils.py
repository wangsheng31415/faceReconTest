import face_recognition
import cv2

def get_face_encodings(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    return face_recognition.face_encodings(rgb_img, face_locations)