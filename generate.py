import torch
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline


# Rassom ismi va janr asosida matnli tavsif (prompt) yaratish funksiyasi
def tavsif_yarat(ism, janr):
    return f"{ism} ismli rassomning {janr} janridagi uslubida, hozirgi zamonni tasvirlovchi zamonaviy rasm."


# Kiritilgan rasmni o‘qish va o‘lchamini 512x512 qilish
def rasm_yuklash(rasm_manzili):
    o_zgartirish = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    rasm = Image.open(rasm_manzili).convert("RGB")
    return o_zgartirish(rasm).unsqueeze(0)


# Tavsif asosida yangi rasmni generatsiya qilish
def yangi_rasm_yaratish(tavsif):
    print("🧠 Model yuklanmoqda...")
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Model tayyor. Rasm yaratilmoqda...")
    natija = model(tavsif).images[0]
    return natija


# Asosiy funksiya: foydalanuvchi ma’lumotlariga asoslangan yangi rasm yaratadi
def zamonaviy_rasm_yaratish(rassom_ismi, janri, rasm_fayli):
    tavsif = tavsif_yarat(rassom_ismi, janri)
    print(f"📝 Tavsif (prompt): {tavsif}")

    # (Bu misolda rasm yuklansa ham, rasm o‘zgarishga ta’sir qilmaydi)
    _ = rasm_yuklash(rasm_fayli)

    yangi_rasm = yangi_rasm_yaratish(tavsif)

    # Rasmni ekranda ko‘rsatish
    yangi_rasm.show()
    # Rasmni faylga saqlash
    yangi_rasm.save("zamonaviy_rasm_natijasi.png")
    print("📸 Yangi rasm 'zamonaviy_rasm_natijasi.png' sifatida saqlandi.")


# 🔽 Misol uchun ishlatish:
# Van Gogh hozir tirik bo‘lganida qanday rasm chizgan bo‘lardi?
zamonaviy_rasm_yaratish("Vincent van Gogh", "Post-Impressionism", "vangogh.jpg")
