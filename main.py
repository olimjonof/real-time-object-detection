import cv2
import numpy as np
import torch
import face_recognition
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from datetime import datetime

# Talabalar bazasi (oldindan suratga olingan)
known_faces = {}
known_ids = {"olimjonov": 1, "joha": 2}  # Talabalar ID ro'yxati

for name in known_ids:
    image = face_recognition.load_image_file(f"faces/{name}.jpg")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces[name] = encoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# YOLOv8 modelini yuklash
model = YOLO("C:/Users/user/Desktop/Real-time Object Detection/runs/detect/train/weights/best.pt").to(device)
model.fuse()  # Inference tezligini oshirish


cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Kamerani ochib bo‘lmadi"

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Kamera kadrni o‘qiy olmadi.")
            break

        # YOLO modelidan foydalangan holda obyektlarni aniqlash
        results = model(frame, conf=0.5, iou=0.5)[0]

        persons = []
        phones = []

        for box in results.boxes:
            cls = int(box.cls)
            name = model.model.names[cls]
            if name == "person":
                persons.append(box.xyxy[0])
            elif name == "cell phone":
                phones.append(box.xyxy[0])

        annotator = Annotator(frame, line_width=2)
        phone_detected = 0
        detected_ids = set()

        # Telefon va odam bog‘lanishini tekshirish
        for phone in phones:
            x1_p, y1_p, x2_p, y2_p = phone
            phone_center = ((x1_p + x2_p) / 2, (y1_p + y2_p) / 2)

            for person in persons:
                x1, y1, x2, y2 = person
                if x1 < phone_center[0] < x2 and y1 < phone_center[1] < y2:
                    phone_detected += 1
                    annotator.box_label(person, label="PHONE DETECTED", color=(0, 0, 255))

        # Talabalarni yuzdan aniqlash
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding, tolerance=0.4)
            name = "Noma'lum"

            if True in matches:
                match_index = matches.index(True)
                name = list(known_faces.keys())[match_index]
                detected_ids.add(known_ids[name])
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Talabalar va telefon ishlatayotganlar sonini chiqarish
        cv2.putText(frame, f"Jami talabalar: {len(detected_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),
                    2)
        cv2.putText(frame, f"Telefon ishlatayotganlar: {phone_detected}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)

        # Foydalanuvchilarni saqlash
        if phone_detected > 0:
            with open("phone_users_log.txt", "a") as f:
                f.write(f"{datetime.now()} - {phone_detected} talaba telefon ishlatdi.\n")

        cv2.imshow("Dars paytida telefon nazorati", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nDastur to‘xtatildi.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Malumotlar tozalandi.")
