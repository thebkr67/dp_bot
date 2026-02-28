import os
import logging
import asyncio
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

from openai import OpenAI


# ---------- setup ----------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=OPENAI_API_KEY)

dp = Dispatcher()


class Reference:
    """
    Stores previous response/context.
    """
    def __init__(self) -> None:
        self.response = ""


reference = Reference()
model_name = "gpt-3.5-turbo"


def clear_past():
    reference.response = ""


# ---------- handlers ----------
@dp.message(Command("start"))
async def welcome(message: Message):
    await message.answer("Hi\nI am here to help you.\nCreated by thebkr")


@dp.message(Command("help"))
async def helper(message: Message):
    help_command = (
        "Hi There, I'm ChatBot created by thebkr. please follow these commands -\n"
        "/start - to start the conversation.\n"
        "/clear - to clear the past conversation and context.\n"
        "/help - to get this help menu.\n\n"
        "I hope this helps. :)"
    )
    await message.answer(help_command)


@dp.message(Command("clear"))
async def clear(message: Message):
    clear_past()
    await message.answer("I've cleared the past conversation and context")


@dp.message()
async def chat_gpt(message: Message):
    user_text = (message.text or "").strip()
    if not user_text:
        return

    logging.info(">>> USER: %s", user_text)

    # IMPORTANT:
    # - role "system" is the correct place for instruction
    # - we keep previous response in context like you did
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Keep responses human-like.\n\n"
                        f"Previous context: {reference.response}"
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        logging.exception("OpenAI request failed")
        await message.answer(f"OpenAI error: {e}")
        return

    reference.response = answer
    logging.info(">>> ChatGPT: %s", answer)

    await message.answer(answer)


# ---------- entrypoint ----------
async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())