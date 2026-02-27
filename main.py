import logging
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
import os
import sys
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


class Reference:
    '''
    A class to store previously response from the chatGPT APi
    '''
    def __init__(self) -> None:
        self.response = ""

def clear_past():
    """
    A function to clear the previous conversation and context.
    """
    reference.response = ""

reference=Reference()
model_name = "gpt-3.5-turbo"



# Initialize bot and dispatcher
bot = Bot(token= TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)



@dp.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """
    This handler receives messages with `/start` command
    """
    await message.reply(f"Hi\nI am here to help you.\nCreated by Asit")


@dp.message_handler(commands=['help'])
async def helper(message: types.Message):
    """
    This handler receives messages with `/start` command
    """
    help_command = """
    Hi There, I'm ChatBot created by Asit. please follow these commands -
    /start - to start the conversation.
    /clear - to clear the past conversation and context.
    /help - to get this help menu.

    I hope this helps. :)
"""
    await message.reply(help_command)


@dp.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """
    This handler to clear the previous conversation and context.
    """
    clear_past()
    await message.reply(f"I've cleared the past conversation and context")


@dp.message_handler()
async def chatGPT(message: types.Message):
    """
    This handler to process the user's input and generate a response using the chatGPT API.
    """
    print(f">>> USER : \n\t{message.text}")
    response = openai.chat.completions.create(
        model = model_name,
        messages = [
            {"role": "assistant", "content":f"you are a romantic boy (the response most be human mostly like a indian boy) flirt with the user in her each and every message: {reference.response} "},
            {"role": "user","content":message.text}
        ]
    )
    reference.response = response.choices[0].message.content
    print(f">>> ChatGPT: \n\t{reference.response}")
    await bot.send_message(chat_id=message.chat.id, text=reference.response)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)