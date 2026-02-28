import os
import io
import re
import json
import base64
import logging
import asyncio
import httpx
from dataclasses import dataclass
from typing import Optional, Tuple

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command

from openai import OpenAI

from pypdf import PdfReader
from docx import Document as DocxDocument
from openpyxl import load_workbook, Workbook
from PIL import Image


# ---------- setup ----------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=OPENAI_API_KEY)
dp = Dispatcher()

TEXT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"

MAX_FILE_BYTES = 12 * 1024 * 1024  # 12 MB: безопасный лимит под Telegram/Railway


class Reference:
    def __init__(self) -> None:
        self.response = ""


reference = Reference()


def clear_past():
    reference.response = ""


# ---------- last file storage ----------
@dataclass
class LastFile:
    filename: str
    ext: str
    mime: str
    data: bytes


last_files: dict[int, LastFile] = {}  # key = telegram user_id


# ---------- helpers ----------
async def serper_search(query: str, num: int = 5) -> list[dict]:
    """
    Возвращает список результатов: title, link, snippet
    """
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is not set")

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": max(1, min(num, 10))}

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    results = []
    for item in (data.get("organic") or [])[:num]:
        results.append({
            "title": item.get("title", ""),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results

def format_search_results(results: list[dict]) -> str:
    text = "🔎 Результаты поиска:\n\n"
    for i, r in enumerate(results, start=1):
        text += f"{i}) {r.get('title','')}\n{r.get('link','')}\n{r.get('snippet','')}\n\n"
    return text.strip()
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
    for page in reader.pages[:max_pages]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    return "\n".join(chunks).strip()


def _extract_text_from_docx(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras).strip()


def _extract_tsv_preview_from_xlsx(data: bytes, max_rows: int = 100, max_cols: int = 25) -> str:
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    rows_out = []
    for r_i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if r_i > max_rows:
            break
        row = row[:max_cols]
        row_s = ["" if v is None else str(v) for v in row]
        rows_out.append("\t".join(row_s).rstrip())
    wb.close()
    return "\n".join(rows_out).strip()


def _extract_text_from_plain(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")


def _detect_image_mime_from_bytes(data: bytes) -> str:
    img = Image.open(io.BytesIO(data))
    fmt = (img.format or "PNG").upper()
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "WEBP":
        return "image/webp"
    return "image/png"


async def _download_telegram_file(bot: Bot, file_id: str) -> Tuple[bytes, str]:
    file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    return buf.getvalue(), (file.file_path or "")


async def _ask_openai_text(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


async def _ask_openai_vision(prompt: str, image_bytes: bytes) -> str:
    mime = _detect_image_mime_from_bytes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

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
    return (r.output_text or "").strip()


def _save_bytes_to_tmp(filename: str, data: bytes) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename)[:120]
    path = f"/tmp/{safe}"
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------- file building ----------
def _build_docx_from_text(text: str) -> bytes:
    doc = DocxDocument()
    # сохраняем абзацы по переносам
    for para in (text or "").splitlines():
        doc.add_paragraph(para)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def _build_xlsx_from_tsv(tsv: str) -> bytes:
    wb = Workbook()
    ws = wb.active
    for r_i, line in enumerate((tsv or "").splitlines(), start=1):
        cols = line.split("\t")
        for c_i, v in enumerate(cols, start=1):
            ws.cell(row=r_i, column=c_i, value=v)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# ---------- edit engines ----------
async def _edit_text_like(original_text: str, instructions: str) -> str:
    system = (
        "Ты редактор. Тебе дают исходный текст и инструкцию, что изменить. "
        "Верни ТОЛЬКО финальную исправленную версию текста без комментариев."
    )
    user = (
        f"ИНСТРУКЦИЯ:\n{instructions}\n\n"
        f"ИСТОЧНИК:\n-----\n{_safe_truncate(original_text)}\n-----\n"
    )
    return await _ask_openai_text(system, user)


async def _edit_docx_bytes(data: bytes, instructions: str) -> Tuple[bytes, str]:
    # Важно: это “умное” редактирование текста, но форматирование docx может упроститься.
    src = _extract_text_from_docx(data)
    edited = await _edit_text_like(src, instructions)
    return _build_docx_from_text(edited), "docx"


async def _edit_xlsx_bytes(data: bytes, instructions: str) -> Tuple[bytes, str]:
    """
    Для xlsx делаем безопасный режим: модель возвращает TSV всей таблицы (первые N строк),
    мы строим новый xlsx. Это надёжнее, чем пытаться “сохранить форматирование”.
    Если нужно сохранить формулы/стили — скажи, добавлю diff-режим по ячейкам.
    """
    preview = _extract_tsv_preview_from_xlsx(data, max_rows=120, max_cols=25)

    system = (
        "Ты ассистент по таблицам. Тебе дают TSV-таблицу (таб-разделители) и инструкцию, что изменить. "
        "Верни ТОЛЬКО итоговый TSV (табами, строки переносами), без пояснений, без markdown."
    )
    user = (
        f"ИНСТРУКЦИЯ:\n{instructions}\n\n"
        f"ТАБЛИЦА TSV:\n-----\n{preview}\n-----\n"
        "Требования: сохраняй структуру таблицы, не добавляй лишних комментариев."
    )
    tsv = await _ask_openai_text(system, user)
    tsv = tsv.strip().strip("```").strip()
    return _build_xlsx_from_tsv(tsv), "xlsx"


# ---------- commands ----------
@dp.message(Command("start"))
async def welcome(message: Message):
    await message.answer(
        "Пришли файл (xlsx/docx/txt/pdf/…) или фото — разберу.\n\n"
        "✅ Редактирование: пришли файл → затем /edit что изменить\n"
        "✅ Создание файлов: /make_xlsx, /make_docx, /make_txt"
    )


@dp.message(Command("help"))
async def helper(message: Message):
    await message.answer(
        "Команды:\n"
        "/start — старт\n"
        "/clear — очистить контекст\n\n"
        "Редактирование файлов:\n"
        "1) отправь файл\n"
        "2) напиши: /edit <инструкция>\n\n"
        "Создание файлов:\n"
        "/make_xlsx <что должно быть в таблице>\n"
        "/make_docx <что должно быть в документе>\n"
        "/make_txt <что должно быть в тексте>\n"
        "/search <запрос> поиск\n"
        "/research <запрос> поиск и сводка от ИИ\n"
    )


@dp.message(Command("clear"))
async def clear(message: Message):
    clear_past()
    await message.answer("Ок, очистил контекст.")


@dp.message(Command("edit"))
async def edit_last_file(message: Message, bot: Bot):
    user_id = message.from_user.id
    instructions = (message.text or "").replace("/edit", "", 1).strip()
    if not instructions:
        await message.answer("Напиши инструкцию так: /edit что именно поменять в последнем файле")
        return

    lf = last_files.get(user_id)
    if not lf:
        await message.answer("Сначала пришли файл, который нужно изменить.")
        return

    if len(lf.data) > MAX_FILE_BYTES:
        await message.answer("Файл слишком большой. Пришли поменьше (до ~12MB).")
        return

    ext = lf.ext
    try:
        if ext in {".txt", ".csv", ".json", ".md", ".log"}:
            src = _extract_text_from_plain(lf.data)
            edited = await _edit_text_like(src, instructions)
            out_bytes = edited.encode("utf-8")
            out_name = f"edited_{re.sub(r'[^a-zA-Z0-9._-]+','_', lf.filename)}"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(FSInputFile(path), caption="Готово. Вот изменённый файл.")
            return

        if ext == ".docx":
            out_bytes, _ = await _edit_docx_bytes(lf.data, instructions)
            out_name = f"edited_{os.path.splitext(lf.filename)[0]}.docx"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(FSInputFile(path), caption="Готово. Вот изменённый DOCX.")
            return

        if ext in {".xlsx", ".xlsm"}:
            out_bytes, _ = await _edit_xlsx_bytes(lf.data, instructions)
            out_name = f"edited_{os.path.splitext(lf.filename)[0]}.xlsx"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(FSInputFile(path), caption="Готово. Вот изменённый XLSX.")
            return

        if ext == ".pdf":
            # PDF править сложно: отдаём docx с правками
            src = _extract_text_from_pdf(lf.data, max_pages=10)
            edited = await _edit_text_like(src, instructions)
            out_bytes = _build_docx_from_text(edited)
            out_name = f"edited_{os.path.splitext(lf.filename)[0]}.docx"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(
                FSInputFile(path),
                caption="PDF как исходник — сложен для правок. Сделал DOCX-версию с изменениями."
            )
            return

        await message.answer("Этот тип файла пока не умею править. Пришли txt/docx/xlsx/pdf.")
    except Exception as e:
        logging.exception("Edit failed")
        await message.answer(f"Не смог применить правки. Ошибка: {e}")


@dp.message(Command("make_xlsx"))
async def make_xlsx(message: Message):
    prompt = (message.text or "").replace("/make_xlsx", "", 1).strip()
    if not prompt:
        await message.answer("Пример: /make_xlsx Сделай таблицу: SKU, Цена, CTR, CR (10 строк демо)")
        return

    system = (
        "Ты генерируешь таблицы. Верни ТОЛЬКО TSV (табами, строки переносами), без markdown и пояснений. "
        "Первая строка — заголовки."
    )
    user = f"Сгенерируй таблицу по запросу:\n{prompt}"
    tsv = await _ask_openai_text(system, user)
    tsv = tsv.strip().strip("```").strip()

    out_bytes = _build_xlsx_from_tsv(tsv)
    out_name = "generated.xlsx"
    path = _save_bytes_to_tmp(out_name, out_bytes)
    await message.answer_document(FSInputFile(path), caption="Сгенерировал XLSX.")


@dp.message(Command("make_docx"))
async def make_docx(message: Message):
    prompt = (message.text or "").replace("/make_docx", "", 1).strip()
    if not prompt:
        await message.answer("Пример: /make_docx Составь регламент обработки отзывов на WB на 1 страницу")
        return

    system = "Сгенерируй документ по запросу. Верни ТОЛЬКО чистый текст документа, без markdown."
    text = await _ask_openai_text(system, prompt)

    out_bytes = _build_docx_from_text(text)
    out_name = "generated.docx"
    path = _save_bytes_to_tmp(out_name, out_bytes)
    await message.answer_document(FSInputFile(path), caption="Сгенерировал DOCX.")


@dp.message(Command("make_txt"))
async def make_txt(message: Message):
    prompt = (message.text or "").replace("/make_txt", "", 1).strip()
    if not prompt:
        await message.answer("Пример: /make_txt Напиши 10 вариантов оффера для карточки товара")
        return

    system = "Сгенерируй текст по запросу. Верни ТОЛЬКО результат, без пояснений."
    text = await _ask_openai_text(system, prompt)

    out_bytes = (text or "").encode("utf-8")
    out_name = "generated.txt"
    path = _save_bytes_to_tmp(out_name, out_bytes)
    await message.answer_document(FSInputFile(path), caption="Сгенерировал TXT.")
@dp.message(Command("search"))
async def cmd_search(message: Message):
    q = (message.text or "").replace("/search", "", 1).strip()
    if not q:
        await message.answer("Напиши так: /search запрос")
        return

    try:
        results = await serper_search(q, num=5)
    except Exception as e:
        await message.answer(f"Ошибка поиска: {e}")
        return

    if not results:
        await message.answer("Ничего не нашёл. Попробуй переформулировать запрос.")
        return

    await message.answer(format_search_results(results))


@dp.message(Command("research"))
async def cmd_research(message: Message):
    q = (message.text or "").replace("/research", "", 1).strip()
    if not q:
        await message.answer("Напиши так: /research запрос (я найду и сделаю краткий вывод)")
        return

    try:
        results = await serper_search(q, num=5)
    except Exception as e:
        await message.answer(f"Ошибка поиска: {e}")
        return

    if not results:
        await message.answer("Ничего не нашёл. Попробуй переформулировать запрос.")
        return

    # 1) показать ссылки
    listing = format_search_results(results)
    await message.answer(listing)

    # 2) сделать сводку моделью (по сниппетам)
    system = (
        "Ты делаешь краткую сводку по результатам интернет-поиска. "
        "Сначала 3-7 буллетов с выводами, затем 'Что сделать дальше' (3-5 шагов). "
        "Если фактов мало — скажи, чего не хватает."
    )
    user = "Запрос: " + q + "\n\n" + "\n".join(
        [f"- {r['title']}: {r['snippet']} ({r['link']})" for r in results]
    )

    try:
        summary = await _ask_openai_text(system, user)
        await message.answer("🧠 Сводка:\n\n" + summary)
    except Exception as e:
        logging.exception("OpenAI summary failed")
        await message.answer(f"Не смог сделать сводку: {e}")

# ---------- file handlers ----------
@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    photo = message.photo[-1]
    image_bytes, _ = await _download_telegram_file(bot, photo.file_id)

    prompt = (
        "Распознай, что на изображении, и помоги пользователю.\n"
        "Если это документ/скрин — извлеки ключевой текст, найди ошибки/суть и дай рекомендации.\n"
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
    mime = doc.mime_type or ""

    file_bytes, _ = await _download_telegram_file(bot, doc.file_id)

    if len(file_bytes) > MAX_FILE_BYTES:
        await message.answer("Файл слишком большой. Пришли поменьше (до ~12MB).")
        return

    # запоминаем как последний файл пользователя
    last_files[message.from_user.id] = LastFile(
        filename=filename,
        ext=ext,
        mime=mime,
        data=file_bytes,
    )

    # Если caption начинается с "edit:" — сразу применяем правки
    caption = (message.caption or "").strip()
    if caption.lower().startswith("edit:"):
        instructions = caption[5:].strip()
        message.text = f"/edit {instructions}"
        await edit_last_file(message, bot)
        return

    # Иначе — просто разбор файла (как у тебя было)
    # 1) Если это картинка — vision
    if ext in {".png", ".jpg", ".jpeg", ".webp"} or mime.startswith("image/"):
        prompt = f"Пользователь прислал изображение файлом: {filename}. Распознай и помоги."
        try:
            answer = await _ask_openai_vision(prompt, file_bytes)
        except Exception as e:
            logging.exception("OpenAI vision failed")
            await message.answer(f"OpenAI error: {e}")
            return
        reference.response = answer
        await message.answer(answer)
        await message.answer("Файл запомнил. Если нужно изменить — напиши /edit <что поменять>.")
        return

    # 2) Извлекаем текст/превью
    try:
        if ext == ".pdf":
            extracted = _extract_text_from_pdf(file_bytes, max_pages=10)
        elif ext == ".docx":
            extracted = _extract_text_from_docx(file_bytes)
        elif ext in {".xlsx", ".xlsm"}:
            extracted = _extract_tsv_preview_from_xlsx(file_bytes, max_rows=120, max_cols=25)
        else:
            extracted = _extract_text_from_plain(file_bytes)
    except Exception as e:
        logging.exception("File parse failed")
        await message.answer(f"Не смог прочитать файл {filename}. Ошибка: {e}")
        return

    extracted = _safe_truncate(extracted, max_chars=45_000)

    user_hint = caption if caption else "Разбери файл: что в нём важного, ошибки/риски, и что делать дальше."
    system = (
        "Отвечай по делу. Если это таблица — дай выводы и рекомендации. "
        "Если это документ — резюме и конкретные улучшения."
    )
    user = (
        f"Файл: {filename}\n"
        f"Задача: {user_hint}\n\n"
        f"Содержимое (может быть обрезано):\n-----\n{extracted}\n-----\n"
    )

    try:
        answer = await _ask_openai_text(system, user)
    except Exception as e:
        logging.exception("OpenAI text failed")
        await message.answer(f"OpenAI error: {e}")
        return

    reference.response = answer
    await message.answer(answer)
    await message.answer("Файл запомнил. Если нужно изменить — напиши /edit <что поменять>.")


@dp.message(F.text)
async def chat_gpt(message: Message):
    user_text = (message.text or "").strip()
    if not user_text:
        return

    try:
        answer = await _ask_openai_text(
            "Отвечай по делу, человеческим языком.",
            user_text
        )
    except Exception as e:
        logging.exception("OpenAI request failed")
        await message.answer(f"OpenAI error: {e}")
        return

    reference.response = answer
    await message.answer(answer)


# ---------- entrypoint ----------
async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())