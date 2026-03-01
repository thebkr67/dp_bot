import os
import io
import re
import json
import base64
import logging
import asyncio
import httpx
import time
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

DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")
DEEPAI_TIMEOUT_SEC = int(os.getenv("DEEPAI_TIMEOUT_SEC", "120"))

if not DEEPAI_API_KEY:
    raise RuntimeError("DEEPAI_API_KEY is not set")


if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=OPENAI_API_KEY)
dp = Dispatcher()

TEXT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"

MAX_FILE_BYTES = 12 * 1024 * 1024  # 12 MB

# ---------- autosearch settings ----------
SEARCH_TRIGGERS = [
    "найди", "поищи", "поиск", "в интернете", "гугл", "google",
    "ссылк", "источник", "пруф", "докажи", "где написано",
    "актуаль", "сейчас", "на сегодня", "последн", "свеж",
    "новост", "цена", "стоимость", "тариф", "курс", "ставк",
    "правила", "регламент", "инструкция", "политика", "обновил",
    "railway", "serper", "aiogram", "wildberries", "ozon"
]

SEARCH_MIN_LEN = 18               # минимальная длина сообщения для автопоиска
SEARCH_RESULTS_NUM = 5            # сколько результатов подмешивать
SEARCH_COOLDOWN_SEC = 12          # 1 поиск / 12 сек на пользователя
SEARCH_CACHE_TTL_SEC = 300        # кэш выдачи на 5 минут


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

# ---------- last image storage ----------
last_images: dict[int, bytes] = {}  # key = telegram user_id


# ---------- autosearch state ----------
_last_search_at: dict[int, float] = {}
_search_cache: dict[str, tuple[float, list[dict]]] = {}


def should_autosearch(text: str) -> bool:
    """
    Решаем, нужен ли автопоиск.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.startswith("/"):
        return False  # команды не трогаем
    if len(t) < SEARCH_MIN_LEN:
        return False
    return any(k in t for k in SEARCH_TRIGGERS)


def can_search_now(user_id: int) -> bool:
    now = time.time()
    last = _last_search_at.get(user_id, 0.0)
    if now - last < SEARCH_COOLDOWN_SEC:
        return False
    _last_search_at[user_id] = now
    return True


def _cache_key(query: str, num: int) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    return f"{q}::num={num}"


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

    async with httpx.AsyncClient(timeout=20) as client_http:
        r = await client_http.post(url, headers=headers, json=payload)
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


async def serper_search_cached(query: str, num: int = 5) -> list[dict]:
    """
    Кэшируем поиск, чтобы не жечь лимиты.
    """
    key = _cache_key(query, num)
    now = time.time()

    if key in _search_cache:
        ts, cached = _search_cache[key]
        if now - ts < SEARCH_CACHE_TTL_SEC:
            return cached

    results = await serper_search(query, num=num)
    _search_cache[key] = (now, results)
    return results


def format_search_results(results: list[dict]) -> str:
    text = "🔎 Результаты поиска:\n\n"
    for i, r in enumerate(results, start=1):
        text += f"{i}) {r.get('title','')}\n{r.get('link','')}\n{r.get('snippet','')}\n\n"
    return text.strip()


def format_results_for_prompt(results: list[dict]) -> str:
    # компактный блок для промпта модели
    lines = []
    for r in results:
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        link = (r.get("link") or "").strip()
        if title or snippet or link:
            lines.append(f"- {title}: {snippet} ({link})")
    return "\n".join(lines).strip()


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
    src = _extract_text_from_docx(data)
    edited = await _edit_text_like(src, instructions)
    return _build_docx_from_text(edited), "docx"


async def _edit_xlsx_bytes(data: bytes, instructions: str) -> Tuple[bytes, str]:
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



# ---------- DeepAI image generation / enhance ----------

DEEPAI_TEXT2IMG_URL = "https://api.deepai.org/api/text2img"
DEEPAI_UPSCALE_URL = "https://api.deepai.org/api/torch-srgan"


async def generate_image_from_text(prompt: str) -> bytes:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Empty prompt")

    files = {"text": (None, prompt)}
    headers = {"api-key": DEEPAI_API_KEY}

    async with httpx.AsyncClient(timeout=DEEPAI_TIMEOUT_SEC) as client_http:
        r = await client_http.post(DEEPAI_TEXT2IMG_URL, headers=headers, files=files)
        r.raise_for_status()
        data = r.json()

    output_url = data.get("output_url")
    if not output_url:
        raise RuntimeError(f"DeepAI error: {data}")

    async with httpx.AsyncClient(timeout=DEEPAI_TIMEOUT_SEC) as client_http:
        img = await client_http.get(output_url)
        img.raise_for_status()
        return img.content


async def enhance_image(image_bytes: bytes) -> bytes:
    if not image_bytes:
        raise ValueError("No image")

    headers = {"api-key": DEEPAI_API_KEY}
    files = {"image": ("input.jpg", image_bytes, "image/jpeg")}

    async with httpx.AsyncClient(timeout=DEEPAI_TIMEOUT_SEC) as client_http:
        r = await client_http.post(DEEPAI_UPSCALE_URL, headers=headers, files=files)
        r.raise_for_status()
        data = r.json()

    output_url = data.get("output_url")
    if not output_url:
        raise RuntimeError(f"DeepAI error: {data}")

    async with httpx.AsyncClient(timeout=DEEPAI_TIMEOUT_SEC) as client_http:
        img = await client_http.get(output_url)
        img.raise_for_status()
        return img.content


@dp.message(Command("img"))
async def cmd_img(message: Message):
    prompt = (message.text or "").replace("/img", "", 1).strip()
    if not prompt:
        await message.answer("Напиши так: /img описание картинки")
        return

    try:
        img_bytes = await generate_image_from_text(prompt)
        path = _save_bytes_to_tmp(f"img_{int(time.time())}.png", img_bytes)
        await message.answer_document(FSInputFile(path), caption="Готово ✅")
    except Exception as e:
        logging.exception("DeepAI img failed")
        await message.answer(f"Ошибка генерации: {e}")


@dp.message(Command("enhance"))
async def cmd_enhance(message: Message):
    user_id = message.from_user.id
    src = last_images.get(user_id)

    if not src:
        await message.answer("Сначала пришли фото, которое нужно улучшить.")
        return

    try:
        img_bytes = await enhance_image(src)
        path = _save_bytes_to_tmp(f"enhanced_{int(time.time())}.png", img_bytes)
        await message.answer_document(FSInputFile(path), caption="Улучшил ✅")
    except Exception as e:
        logging.exception("DeepAI enhance failed")
        await message.answer(f"Ошибка улучшения: {e}")

# ---------- commands ----------
@dp.message(Command("start"))
async def welcome(message: Message):
    await message.answer(
        "Привет, я Альтератти, бот, созданный thebkr.\n"
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
        "/make_txt <что должно быть в тексте>\n\n"
        "Интернет-поиск без команд:\n"
        "Напиши в обычном сообщении: 'найди …', 'дай ссылки …', 'что сейчас …', 'актуальные правила …' — я сам поищу.\n\n"
        "Команды поиска (если всё же надо): /search и /research"
    )


@dp.message(Command("clear"))
async def clear(message: Message):
    clear_past()
    await message.answer("Ок, очистил контекст.")


# --- твои /edit /make_* /search /research и обработчики файлов ниже ОСТАЮТСЯ ---
# (Я их не вырезал — они у тебя уже есть.)


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
        results = await serper_search_cached(q, num=SEARCH_RESULTS_NUM)
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
        results = await serper_search_cached(q, num=SEARCH_RESULTS_NUM)
    except Exception as e:
        await message.answer(f"Ошибка поиска: {e}")
        return

    if not results:
        await message.answer("Ничего не нашёл. Попробуй переформулировать запрос.")
        return

    listing = format_search_results(results)
    await message.answer(listing)

    system = (
        "Ты делаешь краткую сводку по результатам интернет-поиска. "
        "Сначала 3-7 буллетов с выводами, затем 'Что сделать дальше' (3-5 шагов). "
        "Если фактов мало — скажи, чего не хватает."
    )
    user = "Запрос: " + q + "\n\n" + format_results_for_prompt(results)

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

    last_files[message.from_user.id] = LastFile(
        filename=filename,
        ext=ext,
        mime=mime,
        data=file_bytes,
    )

    caption = (message.caption or "").strip()
    if caption.lower().startswith("edit:"):
        instructions = caption[5:].strip()
        message.text = f"/edit {instructions}"
        await edit_last_file(message, bot)
        return

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

    # Команды не трогаем
    if user_text.startswith("/"):
        return

    # --- AUT0SEARCH ---
    search_block = ""
    used_search = False

    if SERPER_API_KEY and should_autosearch(user_text) and can_search_now(message.from_user.id):
        try:
            results = await serper_search_cached(user_text, num=SEARCH_RESULTS_NUM)
            if results:
                used_search = True
                search_block = (
                    "\n\nРЕЗУЛЬТАТЫ ПОИСКА (snippets):\n"
                    + format_results_for_prompt(results)
                    + "\n\nТребование: используй результаты поиска, добавь 2–5 ссылок."
                )
        except Exception:
            logging.exception("Autosearch failed")
            # падаем обратно на обычный ответ без поиска

    system = (
        "Отвечай по делу, человеческим языком. "
        "Если есть блок 'РЕЗУЛЬТАТЫ ПОИСКА' — опирайся на него и добавь ссылки. "
        "Если данных недостаточно — скажи, чего не хватает и что уточнить."
    )

    # подмешиваем контекст + результаты поиска
    user = user_text + search_block

    try:
        answer = await _ask_openai_text(system, user)
    except Exception as e:
        logging.exception("OpenAI request failed")
        await message.answer(f"OpenAI error: {e}")
        return

    reference.response = answer

    # (опционально) маленький индикатор, что поиск применён
    if used_search:
        await message.answer("🌐 Нашёл в интернете, вот по сути:\n\n" + answer)
    else:
        await message.answer(answer)


# ---------- entrypoint ----------
async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())