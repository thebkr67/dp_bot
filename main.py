import os
import io
import re
import base64
import logging
import asyncio
from typing import Optional, Tuple

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command

from openai import OpenAI

from pypdf import PdfReader
from docx import Document as DocxDocument
from openpyxl import load_workbook
from PIL import Image


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
    """Stores previous response/context."""
    def __init__(self) -> None:
        self.response = ""


reference = Reference()

# Текстовая модель (для обычных сообщений и extracted text)
TEXT_MODEL = "gpt-4o-mini"  # можно оставить так, или сменить на то, что у тебя доступно
# Визуальная модель (картинки)
VISION_MODEL = "gpt-4o-mini"


def clear_past():
    reference.response = ""


# ---------- helpers ----------
def _ext(filename: Optional[str]) -> str:
    if not filename:
        return ""
    m = re.search(r"(\.[a-zA-Z0-9]+)$", filename)
    return (m.group(1) if m else "").lower()


def _safe_truncate(s: str, max_chars: int = 45_000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n...[обрезано]..."


def _extract_text_from_pdf(data: bytes, max_pages: int = 10) -> str:
    reader = PdfReader(io.BytesIO(data))
    chunks = []
    for i, page in enumerate(reader.pages[:max_pages]):
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    return "\n".join(chunks).strip()


def _extract_text_from_docx(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras).strip()


def _extract_text_from_xlsx(data: bytes, max_rows: int = 200, max_cols: int = 30) -> str:
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    rows_out = []
    for r_i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if r_i > max_rows:
            break
        row = row[:max_cols]
        # Приводим к строкам
        row_s = ["" if v is None else str(v) for v in row]
        rows_out.append("\t".join(row_s).rstrip())
    wb.close()
    return "\n".join(rows_out).strip()


def _extract_text_from_plain(data: bytes) -> str:
    # пробуем utf-8, потом fallback
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")


def _detect_image_mime_from_bytes(data: bytes) -> str:
    # Pillow умеет определить формат
    img = Image.open(io.BytesIO(data))
    fmt = (img.format or "PNG").upper()
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "WEBP":
        return "image/webp"
    return "image/png"


async def _ask_openai_text(user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Отвечай по делу, человеческим языком. "
                    "Если пользователь прислал текст из файла — сначала кратко резюмируй, "
                    "потом дай выводы/рекомендации.\n\n"
                    f"Previous context: {reference.response}"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


async def _ask_openai_vision(prompt: str, image_bytes: bytes) -> str:
    mime = _detect_image_mime_from_bytes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    # Responses API — универсально для текста+картинки
    r = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )
    # Текст ответа
    try:
        return (r.output_text or "").strip()
    except Exception:
        return "Не смог извлечь текст ответа из результата OpenAI."


async def _download_telegram_file(bot: Bot, file_id: str) -> Tuple[bytes, str]:
    file = await bot.get_file(file_id)
    # В aiogram v3: bot.download_file возвращает BytesIO при destination=BytesIO
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    return buf.getvalue(), (file.file_path or "")


# ---------- commands ----------
@dp.message(Command("start"))
async def welcome(message: Message):
    await message.answer("Добрый день, я Альтереччи. Пришли текст, файл или картинку — разберу и отвечу.")


@dp.message(Command("help"))
async def helper(message: Message):
    await message.answer(
        "Команды:\n"
        "/start — старт\n"
        "/clear — очистить контекст\n\n"
        "Также можно прислать файл (pdf/docx/xlsx/txt/…) или фото — я распознаю и помогу."
    )


@dp.message(Command("clear"))
async def clear(message: Message):
    clear_past()
    await message.answer("Ок, очистил контекст.")


# ---------- file handlers ----------
@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    # Берём самое большое фото
    photo = message.photo[-1]
    image_bytes, _ = await _download_telegram_file(bot, photo.file_id)

    prompt = (
        "Распознай, что на изображении, и помоги пользователю.\n"
        "Если это документ/скрин — извлеки ключевой текст, найди ошибки/суть и дай рекомендации.\n\n"
        f"Previous context: {reference.response}"
    )
    try:
        answer = await _ask_openai_vision(prompt, image_bytes)
    except Exception as e:
        logging.exception("OpenAI vision failed")
        await message.answer(f"OpenAI error: {e}")
        return

    reference.response = answer
    await message.answer(answer)


@dp.message(F.document)
async def handle_document(message: Message, bot: Bot):
    doc = message.document
    filename = doc.file_name or "file"
    ext = _ext(filename)

    file_bytes, _ = await _download_telegram_file(bot, doc.file_id)

    # 1) Если это картинка, отправим в vision
    if ext in {".png", ".jpg", ".jpeg", ".webp"} or (doc.mime_type or "").startswith("image/"):
        prompt = (
            f"Пользователь прислал изображение файлом: {filename}.\n"
            "Распознай содержимое и помоги по задаче пользователя. "
            "Если это скрин/документ — вытащи суть и дай рекомендации.\n\n"
            f"Previous context: {reference.response}"
        )
        try:
            answer = await _ask_openai_vision(prompt, file_bytes)
        except Exception as e:
            logging.exception("OpenAI vision failed")
            await message.answer(f"OpenAI error: {e}")
            return

        reference.response = answer
        await message.answer(answer)
        return

    # 2) Иначе пробуем извлечь текст
    try:
        if ext == ".pdf":
            extracted = _extract_text_from_pdf(file_bytes, max_pages=10)
        elif ext == ".docx":
            extracted = _extract_text_from_docx(file_bytes)
        elif ext in {".xlsx", ".xlsm"}:
            extracted = _extract_text_from_xlsx(file_bytes)
        else:
            # txt/csv/json/log/md + fallback
            extracted = _extract_text_from_plain(file_bytes)
    except Exception as e:
        logging.exception("File parse failed")
        await message.answer(f"Не смог прочитать файл {filename}. Ошибка: {e}")
        return

    extracted = _safe_truncate(extracted, max_chars=45_000)

    user_hint = (message.caption or "").strip()
    if not user_hint:
        user_hint = "Разбери файл: что в нём важного, ошибки/риски, и что делать дальше."

    prompt = (
        f"Пользователь прислал файл: {filename}.\n"
        f"Задача пользователя: {user_hint}\n\n"
        "Текст/содержимое из файла ниже (может быть обрезано):\n"
        "-----\n"
        f"{extracted}\n"
        "-----\n\n"
        "Сначала: краткое резюме (3–7 пунктов). Затем: выводы и конкретные рекомендации."
        "\n\n"
        f"Previous context: {reference.response}"
    )

    try:
        answer = await _ask_openai_text(prompt)
    except Exception as e:
        logging.exception("OpenAI text failed")
        await message.answer(f"OpenAI error: {e}")
        return

    reference.response = answer
    await message.answer(answer)


# ---------- text handler ----------
@dp.message(F.text)
async def chat_gpt(message: Message):
    user_text = (message.text or "").strip()
    if not user_text:
        return

    logging.info(">>> USER: %s", user_text)

    try:
        answer = await _ask_openai_text(user_text)
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