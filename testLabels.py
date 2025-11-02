import os

default_annotation = "0 0.5 0.5 0.1 0.1"

image_folder = "datasets/dataset/images/train"
label_folder = "datasets/dataset/labels/train"

# Labels papkasini yaratish
os.makedirs(label_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Har bir rasm uchun label fayl yaratish
for image_file in image_files:
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(label_folder, label_file)

    # Agar label fayl mavjud bo‘lmasa, default annotatsiya yozish
    if not os.path.exists(label_path):
        with open(label_path, "w") as f:
            f.write(default_annotation + "\n")

print(f"{len(image_files)} ta label fayl yozildi (default annotatsiya).")
