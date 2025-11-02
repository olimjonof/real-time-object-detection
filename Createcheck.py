import cv2
import os
import face_recognition
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime

# === Datasetdan yuzlarni yuklash ===
data_folder = "face_id_dataset"
known_faces = {}
known_ids = {}

print("⏳ Yuz ma'lumotlari yuklanmoqda...")
for person_name in os.listdir(data_folder):
    person_folder = os.path.join(data_folder, person_name)

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_faces[person_name] = encodings[0]
            known_ids[person_name] = len(known_ids) + 1

print(f"✅ {len(known_faces)} ta talaba ma'lumotlari yuklandi!")

# === YOLO modelini yuklash ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolov11x.pt").to(device)
if hasattr(model, "fuse"):
    model.fuse()

# === Kamera ochish ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("▶ Dastur ishga tushdi. Chiqish uchun 'q' tugmasini bosing.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # === YOLO bilan telefon va odamni aniqlash ===
        results = model(frame, conf=0.6, iou=0.5)[0]
        persons = []
        phones = []

        for box in results.boxes:
            cls = int(box.cls)
            name = model.model.names[cls]

            if name == "person":
                persons.append(box.xyxy[0])
            elif name == "cell phone":
                phones.append(box.xyxy[0])

        # === Yuzni aniqlash va identifikatsiya qilish ===
        face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding, tolerance=0.5)
            name = "Noma'lum"
            if True in matches:
                match_index = matches.index(True)
                name = list(known_faces.keys())[match_index]
            face_names.append((name, (top, right, bottom, left)))

        # === Telefon ishlatayotgan odamni aniqlash va kimligini topish ===
        for phone in phones:
            x1_p, y1_p, x2_p, y2_p = phone
            phone_center = ((x1_p + x2_p) / 2, (y1_p + y2_p) / 2)

            for person in persons:
                x1, y1, x2, y2 = person
                if x1 < phone_center[0] < x2 and y1 < phone_center[1] < y2:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, "Telefon ishlatayabdi!!!", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Yuz joylashuvi mos bo‘lsa, ismni aniqlaymiz
                    detected_name = "Noma'lum"
                    for name, (top, right, bottom, left) in face_names:
                        face_center_x = (left + right) / 2
                        face_center_y = (top + bottom) / 2
                        if x1 < face_center_x < x2 and y1 < face_center_y < y2:
                            detected_name = name
                            break

                    # Logga yozish
                    with open("log.txt", "a") as f:
                        f.write(f"{datetime.now()} - {detected_name} telefon ishlatayotgani aniqlandi\n")

        # === Yuzlarni ko‘rsatish ===
        for name, (top, right, bottom, left) in face_names:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Yuz va Telefon Nazorati", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏹ Dastur to‘xtatildi.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Dastur yakunlandi.")