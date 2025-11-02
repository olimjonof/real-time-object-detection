# import cv2
# import os
# import face_recognition
# import numpy as np
#
# # === Dataset uchun saqlash papkasi ===
# data_folder = "dataset_faces"
# os.makedirs(data_folder, exist_ok=True)
#
# # === Kamera ochiladi ===
# cap = cv2.VideoCapture(0)
# assert cap.isOpened(), "Kamera ochilmadi!"
#
# face_count = 0  # Saqlangan yuzlar soni
# face_names = set()  # Takrorlanmas talabalar
#
# def save_face(image, name, face_id):
#     folder_path = os.path.join(data_folder, name)
#     os.makedirs(folder_path, exist_ok=True)
#     filename = f"{folder_path}/{face_id}.jpg"
#     cv2.imwrite(filename, image)
#     print(f"✅ {filename} saqlandi")
#
# try:
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame, model='hog')
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#         for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
#             face_image = frame[top:bottom, left:right]
#             face_id = f"olimjonov_{face_count}"
#             face_count += 1
#
#             # === Agar talaba oldin kiritilmagan bo‘lsa ===
#             if face_id not in face_names:
#                 save_face(face_image, "talaba", face_id)
#                 face_names.add(face_id)
#
#             # === Yuzga bounding box chizish ===
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.putText(frame, "Talaba", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         cv2.imshow("Talaba Yuzlarini Yig'ish", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# except KeyboardInterrupt:
#     print("\nDastur to‘xtatildi.")
#
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
#     print("📁 Dataset yaratish yakunlandi!")

import cv2
import os
import dlib
import face_recognition
import numpy as np

# === Datasetni saqlash uchun ===
data_folder = "face_id_dataset"
os.makedirs(data_folder, exist_ok=True)

# === Kamera ochiladi ===
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Kamera ochilmadi!"

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("C:/Users/user/Desktop/Real-time Object Detection/shape_predictor_68_face_landmarks.dat")


face_count = 0
face_names = set()

def save_face(image, name, face_id):
    folder_path = os.path.join(data_folder, name)
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{folder_path}/{face_id}.jpg"
    cv2.imwrite(filename, image)
    print(f"✅ {filename} saqlandi")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_image = frame[y:y+h, x:x+w]
            face_id = f"olimjonov_{face_count}"
            face_count += 1

            if face_id not in face_names:
                save_face(face_image, "talaba", face_id)
                face_names.add(face_id)

            # === Yuz landshaftlarini aniqlash ===
            landmarks = landmark_predictor(gray, face)
            for i in range(68):
                x_l, y_l = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(frame, (x_l, y_l), 2, (0, 255, 0), -1)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Talaba", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face ID Yuzni Tanish", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nDastur to‘xtatildi.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("📁 Dataset yaratish yakunlandi!")
