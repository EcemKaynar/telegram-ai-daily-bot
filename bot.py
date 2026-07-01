import os
import telebot
from dotenv import load_dotenv
from openrouter_client import ask_openrouter

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN bulunamadı.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(message, "Merhaba! Ben günlük sorularını cevaplamak için hazırlanan bir Telegram botum. Lütfen sorularınızı sorunuz.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text

    bot.send_chat_action(message.chat.id, "typing")

    ai_answer = ask_openrouter(user_text)

    bot.reply_to(message, ai_answer)


print("Bot başlatıldı.")
bot.infinity_polling()