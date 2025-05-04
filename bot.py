import torch
import torch.nn as nn
import timm  # используем ту же библиотеку, что и при обучении
from torchvision import transforms
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import io

# === Настройки модели ===
NUM_CLASSES = 25
MODEL_PATH = "efficientnet_finetuned.pth"

# === Проверка устройства ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Загрузка модели через timm ===
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Преобразования ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Названия классов ===
idx_to_class = {
    0: "Achaemenid architecture", 1: "American Foursquare architecture", 2: "American craftsman style",
    3: "Ancient Egyptian architecture", 4: "Art Deco architecture", 5: "Art Nouveau architecture",
    6: "Baroque architecture", 7: "Bauhaus architecture", 8: "Beaux-Arts architecture",
    9: "Byzantine architecture", 10: "Chicago school architecture", 11: "Colonial architecture",
    12: "Deconstructivism", 13: "Edwardian architecture", 14: "Georgian architecture",
    15: "Gothic architecture", 16: "Greek Revival architecture", 17: "International style",
    18: "Novelty architecture", 19: "Palladian architecture", 20: "Postmodern architecture",
    21: "Queen Anne architecture", 22: "Romanesque architecture", 23: "Russian Revival architecture",
    24: "Tudor Revival architecture"
}

# === Команды ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Отправь мне фото здания, и я определю архитектурный стиль 🏛️")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_stream = io.BytesIO()
    await file.download_to_memory(out=image_stream)
    image_stream.seek(0)

    image = Image.open(image_stream).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class = idx_to_class.get(predicted.item(), "Неизвестно")
    await update.message.reply_text(f"📸 Предсказанный архитектурный стиль: {predicted_class}")

# === Запуск ===
def main():
    TOKEN = "7854664139:AAFK2DtlUgK8jO1sg7wyvPA8mtEghucJDSA"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    app.run_polling()

if __name__ == '__main__':
    main()
