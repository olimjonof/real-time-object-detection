
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime

# GPU yoki CPU aniqlash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv11 modelini yuklash va optimallashtirish
model = YOLO("C:/Users/user/Desktop/Real-time Object Detection/yolo11x.pt").to(device)

if hasattr(model, "fuse"):
    model.fuse()  # Inference tezligini oshirish uchun

names = model.model.names
needed_classes = ["person", "cell phone"]

# Kamera ochish
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Kamera ishlamasa, chiqish

    # Model bilan obyektlarni aniqlash
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

    detected = False  # Telefon ishlatayotgan talabalar borligini tekshirish uchun

    # Odam va telefon orasidagi masofani tekshirish
    for phone in phones:
        x1_p, y1_p, x2_p, y2_p = phone
        phone_center = ((x1_p + x2_p) / 2, (y1_p + y2_p) / 2)

        for person in persons:
            x1, y1, x2, y2 = person

            if x1 < phone_center[0] < x2 and y1 < phone_center[1] < y2:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)  # Qizil chiziq
                cv2.putText(frame, "Telefon ishlatayabdi!!!", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                detected = True

                # Natijalarni log fayliga yozish
                with open("log.txt", "a") as f:
                    f.write(f"{datetime.now()} - Telefon ishlatilayotgan aniqlandi\n")

    cv2.imshow("Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
