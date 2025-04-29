import torch
from torchvision import models, transforms
from PIL import Image
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# === Настройки модели ===
NUM_CLASSES = 25  # замените на своё количество классов
MODEL_PATH = "efficientnet_finetuned.pth"

# === Загрузка модели ===
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# === Преобразования для изображений ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Словарь классов (пример, подставь свои названия) ===
idx_to_class = {
    0: "Achaemenid architecture", 1: "American Foursquare architecture", 2: "American craftsman style", 3: "Ancient Egyptian architecture", 4: "Art Deco architecture",
    5: "Art Nouveau architecture", 6: "Baroque architecture", 7: "Bauhaus architecture", 8: "Beaux-Arts architecture", 9: "Byzantine architecture",
    10: "Chicago school architecture", 11: "Colonial architecture", 12: "Deconstructivism", 13: "Edwardian architecture",
    14: "Georgian architecture", 15: "Gothic architecture", 16: "Greek Revival architecture", 17: "International style",
    18: "Novelty architecture", 19: "Palladian architecture", 20: "Postmodern architecture", 21: "Queen Anne architecture",
    22: "Romanesque architecture", 23: "Russian Revival architecture", 24: "Tudor Revival architecture"
}

# === Предсказание архитектурного стиля ===
def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return idx_to_class.get(predicted.item(), "Неизвестно")

# === Команды Telegram-бота ===
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Привет! Отправь мне фото здания, и я попробую определить архитектурный стиль 🏛️")

def handle_photo(update: Update, context: CallbackContext):
    photo = update.message.photo[-1]
    file = context.bot.get_file(photo.file_id)
    file_path = "received.jpg"
    file.download(file_path)

    predicted_class = classify_image(file_path)
    update.message.reply_text(f"📸 Предсказанный архитектурный стиль: {predicted_class}")

# === Запуск бота ===
def main():
    TOKEN = "7854664139:AAGjNjdmjZPGv6XbqqNnA07x-6aBVfBa9UY"  # Замени на свой токен от BotFather
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
