import logging
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
import os

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# print(TELEGRAM_BOT_TOKEN)

# configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token= TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)



@dp.message_handler(commands=['start','help'])
async def command_start_handler(message: types.Message):
    """
    This handler receives messages with `/start` command
    """
    await message.reply(f"Hi\nI am here to help you.\nPowered By ChatGPT")
@dp.message_handler()
async def echo(message: types.Message):
    """
    This will return the same words
    """
    await message.answer(message.text)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
