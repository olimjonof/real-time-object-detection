import cv2
import numpy as np
import torch
import face_recognition
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from collections import defaultdict

# === YOLO modelini yuklash ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("C:/Users/user/Desktop/Real-time Object Detection/runs/detect/train/weights/best.pt").to(device)

# YOLO modelining sinflarini tekshiramiz
print("Modeldagi sinflar:", model.model.names)

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Kamerani ochib bo‘lmadi"

# Kamera hajmini kichraytirish
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Talabalar bazasi ===
known_faces = {}
known_ids = {"olimjonov": 1, "joha": 2}

for name in known_ids:
    image = face_recognition.load_image_file(f"dataset_faces/{name}.jpg")
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_faces[name] = encoding[0]

# === Talabalar harakatlarini kuzatish uchun vaqt hisobi ===
head_down_time = defaultdict(int)
hand_down_time = defaultdict(int)
phone_detected_time = defaultdict(int)

def send_alert(message):
    print(f"⚠️ OG'OHLANTIRISH: {message}")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        results = model(rgb_frame, conf=0.3, max_det=10)[0]

        persons, hands, phones = [], [], []

        for box in results.boxes:
            cls = int(box.cls)
            name = model.model.names.get(cls, "Unknown")

            if name == "person":
                persons.append(box.xyxy[0])
            elif name == "hand":
                hands.append(box.xyxy[0])
            elif name == "cell phone":
                phones.append(box.xyxy[0])

        annotator = Annotator(frame, line_width=2)

        # === Parta tagiga qarayotgan talabalarni aniqlash ===
        for person in persons:
            x1, y1, x2, y2 = person
            center_y = (y1 + y2) / 2

            if center_y > frame.shape[0] * 0.6:
                head_down_time[x1] += 1
            else:
                head_down_time[x1] = 0

            if head_down_time[x1] > 90:
                annotator.box_label(person, label="SHPARGALKA!!!", color=(0, 0, 255))
                send_alert("Talaba parta tagiga qarayapti!")

        # === Qo‘llari pastda bo‘lganlarni aniqlash ===
        for hand in hands:
            x1, y1, x2, y2 = hand
            if y1 > frame.shape[0] * 0.6:
                hand_down_time[x1] += 1
            else:
                hand_down_time[x1] = 0

            if hand_down_time[x1] > 90:
                annotator.box_label(hand, label="Qo'l pastda!", color=(255, 0, 255))
                send_alert("Talabaning qo‘li 3 sekunddan ko‘p pastda!")

        # === Telefon ishlatayotgan talabalarni aniqlash ===
        for phone in phones:
            x1_p, y1_p, x2_p, y2_p = phone
            phone_center = ((x1_p + x2_p) / 2, (y1_p + y2_p) / 2)

            for person in persons:
                x1, y1, x2, y2 = person
                if x1 < phone_center[0] < x2 and y1 < phone_center[1] < y2:
                    phone_detected_time[x1] += 1
                else:
                    phone_detected_time[x1] = 0

                if phone_detected_time[x1] > 90:
                    annotator.box_label(person, label="Telefon!!!", color=(0, 0, 255))
                    send_alert("Talaba telefon ishlatmoqda!")

        # === Talabalarni yuzdan aniqlash ===
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding, tolerance=0.5)
            name = "Noma'lum"

            if True in matches:
                match_index = matches.index(True)
                name = list(known_faces.keys())[match_index]

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # === Natijalarni ekranga chiqarish ===
        cv2.putText(frame, f"Talabalar: {len(persons)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Telefon ishlatayotganlar: {len(phones)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Imtihon nazorati", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nDastur to‘xtadi.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Malumotlar tozalandi.")
