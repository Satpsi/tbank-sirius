import telebot
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

bot_token = "7601677786:AAG-7JrW_EXnDsiGvSoYSwHH8PSxrxD23kw"
bot = telebot.TeleBot(bot_token)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
last_description = ""

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(
        message.chat.id,
        "Привет! Отправь мне фото, и я опишу одежду на нем"
    )

# Обработчик фото
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    global last_description

    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image = Image.open(io.BytesIO(downloaded_file))
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    last_description = processor.decode(output[0], skip_special_tokens=True)

    markup = telebot.types.InlineKeyboardMarkup()
    markup.add(
        telebot.types.InlineKeyboardButton("Все вещи", callback_data="all_items"),
        telebot.types.InlineKeyboardButton("Похожие вещи", callback_data="similar_items"),
        telebot.types.InlineKeyboardButton("Конкретные вещи", callback_data="specific_items")
    )
    bot.send_message(message.chat.id, "Выберите, что хотите:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def handle_query(call):
    if call.data == "all_items":
        bot.send_message(call.message.chat.id, f'описание что на фото {last_description}')
    elif call.data == "similar_items":
        bot.send_message(call.message.chat.id, "Здесь могут быть показаны похожие вещи (заглушка)")
    elif call.data == "specific_items":
        bot.send_message(call.message.chat.id, "Здесь будет выбор конкретных вещей (заглушка)")

bot.polling(none_stop=True)
