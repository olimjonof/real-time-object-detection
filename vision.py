import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

# Sinflarni o‘zbek tiliga tarjima qilish
uzbek_labels = {
    "person": "odam",
    "bicycle": "velosiped",
    "car": "mashina",
    "motorcycle": "mototsikl",
    "airplane": "samolyot",
    "bus": "avtobus",
    "train": "poyezd",
    "truck": "yuk mashinasi",
    "boat": "qayiq",
    "traffic light": "svetofor",
    "fire hydrant": "gazo‘chir",
    "stop sign": "to‘xtash belgisi",
    "parking meter": "to‘lov avtomati",
    "bench": "skameyka",
    "bird": "qush",
    "cat": "mushuk",
    "dog": "it",
    "horse": "ot",
    "sheep": "qo‘y",
    "cow": "sigir",
    "elephant": "fil",
    "bear": "ayiq",
    "zebra": "zebra",
    "giraffe": "jirafa",
    "backpack": "ryukzak",
    "umbrella": "soyabon",
    "handbag": "sumka",
    "tie": "galstuk",
    "suitcase": "chemodan",
    "frisbee": "disk",
    "skis": "chang‘i",
    "snowboard": "snoubord",
    "sports ball": "sport to‘pi",
    "kite": "laylak",
    "baseball bat": "beysboll tayog‘i",
    "baseball glove": "beysboll qo‘li",
    "skateboard": "skeytbord",
    "surfboard": "serfbord",
    "tennis racket": "tennis raketkasi",
    "bottle": "shisha",
    "wine glass": "vino bokali",
    "cup": "chashka",
    "fork": "sanchqi",
    "knife": "pichoq",
    "spoon": "qoshiq",
    "bowl": "kosacha",
    "banana": "banan",
    "apple": "olma",
    "sandwich": "sendvich",
    "orange": "apelsin",
    "broccoli": "brokkoli",
    "carrot": "sabzi",
    "hot dog": "hot-dog",
    "pizza": "pitsa",
    "donut": "ponchik",
    "cake": "tort",
    "chair": "stul",
    "couch": "divan",
    "potted plant": "guldon",
    "bed": "karavot",
    "dining table": "ovqat stoli",
    "toilet": "hojatxona",
    "tv": "televizor",
    "laptop": "noutbuk",
    "mouse": "sichqoncha",
    "remote": "pult",
    "keyboard": "klaviatura",
    "cell phone": "telefon",
    "microwave": "mikroto‘lqinli pech",
    "oven": "duhovka",
    "toaster": "toster",
    "sink": "rakovina",
    "refrigerator": "muzlatkich",
    "book": "kitob",
    "clock": "soat",
    "vase": "vaza",
    "scissors": "qaychi",
    "teddy bear": "o‘yinchoq ayiq",
    "hair drier": "soch quritgich",
    "toothbrush": "tish cho‘tka"
}

# Initialize tracking history
track_history = defaultdict(lambda: [])

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("models/yolov8n.pt").to(device)

names = model.model.names  # COCO sinflari

# Initialize video capture (default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), "Error opening camera"

try:
    while True:
        # Read frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break

        # Perform object tracking
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy  # Bounding boxes

        if results[0].boxes.id is not None:
            clss = results[0].boxes.cls.tolist()
            track_ids = results[0].boxes.id.int().tolist()

            # Annotator initialization
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                object_name = names[int(cls)]
                uzbek_name = uzbek_labels.get(object_name, object_name)  # O‘zbekcha nomni olish

                # Annotate bounding boxes and labels
                annotator.box_label(
                    box, color=colors(int(cls), True), label=f"{uzbek_name} {track_id}"
                )

                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30:
                    track.pop(0)

                # Plot tracks on the frame
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, track[-1], 7, colors(int(cls), True), -1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

        # Display the frame with annotations
        cv2.imshow("Real-time Object Detection - Uzbek", frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quitting...")
            break

except KeyboardInterrupt:
    print("\nProgram interrupted manually.")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Program terminated.")