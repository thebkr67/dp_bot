import os
import io
import re
import time
import json
import base64
import logging
import asyncio
from dataclasses import dataclass
from typing import Tuple, Optional, List

import httpx
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile, CallbackQuery
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from pypdf import PdfReader
from docx import Document as DocxDocument
from openpyxl import load_workbook, Workbook
from PIL import Image

# ---------- setup ----------
load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# DeepSeek (text)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Serper (optional web search)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# DeepAI (optional image tools)
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")
DEEPAI_TIMEOUT_SEC = int(os.getenv("DEEPAI_TIMEOUT_SEC", "120"))
ENHANCE_DEFAULT_SCALE = int(os.getenv("ENHANCE_DEFAULT_SCALE", "2"))

# fal.ai (optional img2vid)
FAL_KEY = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
FAL_QUEUE_BASE = os.getenv("FAL_QUEUE_BASE", "https://queue.fal.run")
PIKA_IMAGE2VIDEO_MODEL = os.getenv("PIKA_IMAGE2VIDEO_MODEL", "fal-ai/pika/v2.2/image-to-video")
PIKA_POLL_INTERVAL_SEC = int(os.getenv("PIKA_POLL_INTERVAL_SEC", "5"))
PIKA_TASK_TIMEOUT_SEC = int(os.getenv("PIKA_TASK_TIMEOUT_SEC", "600"))
DEFAULT_VIDEO_PROMPT = os.getenv("DEFAULT_VIDEO_PROMPT",
                                  "create a short video from this image, make it come alive with gentle motion")

MAX_FILE_BYTES = 12 * 1024 * 1024  # 12 MB

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY is not set")

dp = Dispatcher()

# ---------- state ----------
class Reference:
    def __init__(self) -> None:
        self.response = ""

reference = Reference()

def clear_past():
    reference.response = ""

@dataclass
class LastFile:
    filename: str
    ext: str
    mime: str
    data: bytes

last_files: dict[int, LastFile] = {}
last_images: dict[int, bytes] = {}

# ---------- utils ----------
def _ext(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()

def _save_bytes_to_tmp(filename: str, data: bytes) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename)[:120]
    path = f"/tmp/{safe}"
    with open(path, "wb") as f:
        f.write(data)
    return path

async def _download_telegram_file(bot: Bot, file_id: str) -> Tuple[bytes, str]:
    file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    return buf.getvalue(), (file.file_path or "")

def _extract_text_from_plain(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1251", errors="replace")

def _extract_text_from_pdf(data: bytes, max_pages: int = 10) -> str:
    reader = PdfReader(io.BytesIO(data))
    out = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        out.append(page.extract_text() or "")
    return "\n".join(out).strip()

def _extract_text_from_docx(data: bytes) -> str:
    doc = DocxDocument(io.BytesIO(data))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras).strip()

def _extract_tsv_preview_from_xlsx(data: bytes, max_rows: int = 120, max_cols: int = 25) -> str:
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

def _build_xlsx_from_tsv(tsv: str) -> bytes:
    wb = Workbook()
    ws = wb.active
    for r_i, line in enumerate((tsv or "").splitlines(), start=1):
        cells = line.split("\t")
        for c_i, val in enumerate(cells, start=1):
            ws.cell(row=r_i, column=c_i, value=val)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

def _build_docx_from_text(text: str) -> bytes:
    doc = DocxDocument()
    for line in (text or "").splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def _safe_truncate(s: str, max_chars: int = 45000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n…(обрезано)…"

# ---------- new helper: split long message ----------
def split_message(text: str, max_len: int = 4000) -> List[str]:
    """
    Разбивает длинный текст на части, не превышающие max_len символов.
    Старается не резать слова (режет по пробелу или переносу строки).
    """
    if len(text) <= max_len:
        return [text]

    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break

        # Ищем место разрыва: последний пробел или перенос в пределах max_len
        split_at = max_len
        # Пытаемся найти пробел или \n в диапазоне от max_len-200 до max_len
        for sep in ('\n', ' '):
            pos = text.rfind(sep, max_len - 200, max_len)
            if pos != -1:
                split_at = pos + 1  # включаем разделитель в текущую часть
                break
        else:
            # Если подходящий разделитель не найден, режем ровно по max_len
            split_at = max_len

        parts.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    return parts

async def _react_ok(message: Message):
    try:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    except Exception:
        pass

async def run_with_thinking(bot: Bot, chat_id: int, coro):
    task = asyncio.create_task(coro)
    while not task.done():
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return await task

# ---------- DeepSeek client ----------
async def _deepseek_chat(system: str, user: str) -> str:
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system or "Ты полезный ассистент."},
            {"role": "user", "content": user or ""},
        ],
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

async def _ask_llm_text(system: str, user: str) -> str:
    return await _deepseek_chat(system, user)

# ---------- Serper search (optional) ----------
async def serper_search(q: str, num: int = 5) -> list[dict]:
    if not SERPER_API_KEY:
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": q, "num": num}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    out = []
    for it in (data.get("organic") or [])[:num]:
        out.append({"title": it.get("title"), "link": it.get("link"), "snippet": it.get("snippet")})
    return out

def format_search_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title") or ""
        link = r.get("link") or ""
        snip = r.get("snippet") or ""
        lines.append(f"{i}. {title}\n{snip}\n{link}")
    return "\n\n".join(lines).strip()

def format_results_for_prompt(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"- {r.get('title','')}: {r.get('snippet','')} ({r.get('link','')})")
    return "\n".join(lines).strip()

# ---------- DeepAI (optional) ----------
DEEPAI_UPSCALE_URL = "https://api.deepai.org/api/torch-srgan"

async def enhance_image_deepai(image_bytes: bytes) -> bytes:
    if not DEEPAI_API_KEY:
        raise RuntimeError("DEEPAI_API_KEY is not set")
    files = {"image": ("image.png", image_bytes)}
    headers = {"api-key": DEEPAI_API_KEY}
    async with httpx.AsyncClient(timeout=DEEPAI_TIMEOUT_SEC) as client:
        r = await client.post(DEEPAI_UPSCALE_URL, headers=headers, files=files)
        r.raise_for_status()
        data = r.json()
    output_url = data.get("output_url")
    if not output_url:
        raise RuntimeError(f"DeepAI error: {data}")
    async with httpx.AsyncClient(timeout=DEEPAI_TIMEOUT_SEC) as client:
        img = await client.get(output_url)
        img.raise_for_status()
        return img.content

async def enhance_image(image_bytes: bytes, scale: int = 2) -> bytes:
    scale = 4 if int(scale) == 4 else 2
    out = await enhance_image_deepai(image_bytes)
    if scale == 4:
        out = await enhance_image_deepai(out)
    return out

# ---------- fal.ai img2vid (optional) ----------
def _fal_headers() -> dict:
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is not set")
    return {"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"}

def _detect_image_mime_from_bytes(data: bytes) -> str:
    img = Image.open(io.BytesIO(data))
    fmt = (img.format or "PNG").upper()
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "WEBP":
        return "image/webp"
    return "image/png"

def _to_data_uri(image_bytes: bytes) -> str:
    mime = _detect_image_mime_from_bytes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

async def fal_queue_submit(model_id: str, payload: dict) -> dict:
    url = f"{FAL_QUEUE_BASE}/{model_id}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=_fal_headers(), json=payload)
        r.raise_for_status()
        return r.json()

async def fal_wait_for_result_by_urls(status_url: str, response_url: str, timeout_sec: int = PIKA_TASK_TIMEOUT_SEC) -> dict:
    start = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            r = await client.get(status_url, headers=_fal_headers())
            if r.status_code not in (200, 202):
                r.raise_for_status()
            status_obj = r.json()
            if status_obj.get("status") == "COMPLETED":
                rr = await client.get(response_url, headers=_fal_headers())
                rr.raise_for_status()
                return rr.json()
            if time.time() - start > timeout_sec:
                raise TimeoutError("fal.ai task timeout")
            await asyncio.sleep(max(1, PIKA_POLL_INTERVAL_SEC))

async def pika_image_bytes_to_video(image_bytes: bytes, prompt: str) -> bytes:
    payload = {"image_url": _to_data_uri(image_bytes), "prompt": prompt}
    submit = await fal_queue_submit(PIKA_IMAGE2VIDEO_MODEL, payload)
    status_url = submit.get("status_url")
    response_url = submit.get("response_url")
    if not status_url or not response_url:
        raise RuntimeError(f"fal.ai submit failed: {submit}")
    result = await fal_wait_for_result_by_urls(status_url, response_url)
    video_url = ((result or {}).get("video") or {}).get("url")
    if not video_url:
        raise RuntimeError(f"fal.ai result missing video url: {result}")
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(video_url)
        r.raise_for_status()
        return r.content

# ---------- keyboards ----------
ENHANCE_CB = "enhance:last"
VIDEO_CB = "video:last"

def _image_action_keyboard():
    kb = InlineKeyboardBuilder()
    kb.button(text="✨ Улучшить", callback_data=ENHANCE_CB)
    kb.button(text="🎥 Видео", callback_data=VIDEO_CB)
    kb.adjust(2)
    return kb.as_markup()

# ---------- commands ----------
@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "Привет! Я бот без OpenAI: текст — через DeepSeek API.\n"
        "Пришли текст/файл/фото.\n\n"
        "Команды:\n"
        "/clear — очистить контекст\n"
        "/search запрос — поиск (если настроен SERPER_API_KEY)\n"
        "/edit <что поменять> — правка последнего файла (txt/docx/xlsx/pdf)\n"
        "/enhance 2|4 — улучшить последнее фото (если настроен DEEPAI_API_KEY)\n"
        "/img2vid <описание> — видео из последнего фото (если настроен FAL_KEY)\n"
    )

@dp.message(Command("clear"))
async def clear(message: Message):
    clear_past()
    await message.answer("Ок, очистил контекст.")

@dp.message(Command("search"))
async def cmd_search(message: Message):
    try:
        q = (message.text or "").replace("/search", "", 1).strip()
        if not q:
            await message.answer("Напиши так: /search запрос")
            return
        if not SERPER_API_KEY:
            await message.answer("SERPER_API_KEY не задан — поиск отключён.")
            return
        results = await serper_search(q, num=5)
        if not results:
            await message.answer("Ничего не нашёл. Попробуй переформулировать.")
            return
        text = format_search_results(results)
        # FIX: split long message
        for part in split_message(text):
            await message.answer(part)
    except Exception as e:
        logging.exception("Search failed")
        await message.answer(f"Ошибка поиска: {e}")

@dp.message(Command("enhance"))
async def cmd_enhance(message: Message):
    try:
        src = last_images.get(message.from_user.id)
        if not src:
            await message.answer("Сначала пришли фото.")
            return
        if not DEEPAI_API_KEY:
            await message.answer("DEEPAI_API_KEY не задан — улучшение отключено.")
            return
        m = re.search(r"\b(2|4)\s*x?\b", (message.text or ""))
        scale = int(m.group(1)) if m else ENHANCE_DEFAULT_SCALE
        await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_DOCUMENT)
        out = await run_with_thinking(message.bot, message.chat.id, enhance_image(src, scale=scale))
        path = _save_bytes_to_tmp(f"enhanced_{int(time.time())}.png", out)
        await message.answer_document(FSInputFile(path), caption="Улучшил ✅")
    except Exception as e:
        logging.exception("Enhance failed")
        await message.answer(f"Ошибка улучшения: {e}")

@dp.message(Command("img2vid"))
async def cmd_img2vid(message: Message):
    try:
        prompt = (message.text or "").replace("/img2vid", "", 1).strip()
        if not prompt:
            await message.answer("Напиши так: /img2vid описание (движение/камера/стиль).")
            return
        src = last_images.get(message.from_user.id)
        if not src:
            await message.answer("Сначала пришли фото.")
            return
        if not FAL_KEY:
            await message.answer("FAL_KEY не задан — видео отключено.")
            return
        await message.bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_VIDEO)
        video_bytes = await run_with_thinking(message.bot, message.chat.id, pika_image_bytes_to_video(src, prompt))
        path = _save_bytes_to_tmp(f"pika_{int(time.time())}.mp4", video_bytes)
        await message.answer_video(FSInputFile(path), caption="Видео готово ✅")
    except Exception as e:
        logging.exception("img2vid failed")
        await message.answer(f"Ошибка генерации видео: {e}")

# ---------- file editing ----------
async def _edit_text_like(src: str, instructions: str) -> str:
    system = (
        "Ты редактор. Прими исходный текст и инструкцию, верни только итоговый текст без пояснений."
    )
    user = f"ИНСТРУКЦИЯ:\n{instructions}\n\nТЕКСТ:\n-----\n{src}\n-----"
    return await _ask_llm_text(system, user)

async def _edit_docx_bytes(data: bytes, instructions: str) -> bytes:
    src = _extract_text_from_docx(data)
    edited = await _edit_text_like(src, instructions)
    return _build_docx_from_text(edited)

async def _edit_xlsx_bytes(data: bytes, instructions: str) -> bytes:
    preview = _extract_tsv_preview_from_xlsx(data, max_rows=120, max_cols=25)
    system = (
        "Ты ассистент по таблицам. Тебе дают TSV и инструкцию. "
        "Верни ТОЛЬКО итоговый TSV (табами, строки переносами), без markdown и пояснений."
    )
    user = (
        f"ИНСТРУКЦИЯ:\n{instructions}\n\n"
        f"ТАБЛИЦА TSV:\n-----\n{preview}\n-----\n"
        "Требования: сохраняй структуру таблицы."
    )
    tsv = (await _ask_llm_text(system, user)).strip().strip("```").strip()
    return _build_xlsx_from_tsv(tsv)

@dp.message(Command("edit"))
async def edit_last_file(message: Message):
    try:
        user_id = message.from_user.id
        instructions = (message.text or "").replace("/edit", "", 1).strip()
        if not instructions:
            await message.answer("Напиши инструкцию так: /edit что именно поменять в последнем файле")
            return
        lf = last_files.get(user_id)
        if not lf:
            await message.answer("Сначала пришли файл, который нужно изменить.")
            return

        if lf.ext in {".txt", ".csv", ".json", ".md", ".log"}:
            src = _extract_text_from_plain(lf.data)
            edited = await _edit_text_like(src, instructions)
            out_bytes = edited.encode("utf-8")
            out_name = f"edited_{re.sub(r'[^a-zA-Z0-9._-]+','_', lf.filename)}"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(FSInputFile(path), caption="Готово. Вот изменённый файл.")
            return

        if lf.ext == ".docx":
            out_bytes = await _edit_docx_bytes(lf.data, instructions)
            out_name = f"edited_{os.path.splitext(lf.filename)[0]}.docx"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(FSInputFile(path), caption="Готово. Вот изменённый DOCX.")
            return

        if lf.ext in {".xlsx", ".xlsm"}:
            out_bytes = await _edit_xlsx_bytes(lf.data, instructions)
            out_name = f"edited_{os.path.splitext(lf.filename)[0]}.xlsx"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(FSInputFile(path), caption="Готово. Вот изменённый XLSX.")
            return

        if lf.ext == ".pdf":
            src = _extract_text_from_pdf(lf.data, max_pages=10)
            edited = await _edit_text_like(src, instructions)
            out_bytes = _build_docx_from_text(edited)
            out_name = f"edited_{os.path.splitext(lf.filename)[0]}.docx"
            path = _save_bytes_to_tmp(out_name, out_bytes)
            await message.answer_document(FSInputFile(path), caption="PDF сложен для правок. Сделал DOCX-версию с изменениями.")
            return

        await message.answer("Этот тип файла пока не умею править. Пришли txt/docx/xlsx/pdf.")
    except Exception as e:
        logging.exception("Edit failed")
        await message.answer(f"Не смог применить правки. Ошибка: {e}")

# ---------- handlers ----------
@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    await _react_ok(message)
    try:
        photo = message.photo[-1]
        image_bytes, _ = await _download_telegram_file(bot, photo.file_id)
    except Exception as e:
        await message.answer(f"Не удалось загрузить фото: {e}")
        return
    last_images[message.from_user.id] = image_bytes

    await message.answer("Фото получил ✅")
    await message.answer("Что сделать с фото?", reply_markup=_image_action_keyboard())

@dp.callback_query(F.data == ENHANCE_CB)
async def cb_enhance_last(callback: CallbackQuery):
    try:
        src = last_images.get(callback.from_user.id)
        if not src:
            await callback.message.answer("Не нашёл последнее фото. Пришли фото ещё раз.")
            await callback.answer()
            return
        if not DEEPAI_API_KEY:
            await callback.message.answer("DEEPAI_API_KEY не задан — улучшение отключено.")
            await callback.answer()
            return
        await callback.answer("Улучшаю…")
        out = await run_with_thinking(callback.message.bot, callback.message.chat.id,
                                      enhance_image(src, scale=ENHANCE_DEFAULT_SCALE))
        path = _save_bytes_to_tmp(f"enhanced_{int(time.time())}.png", out)
        await callback.message.answer_document(FSInputFile(path), caption="Улучшил ✅")
    except Exception as e:
        logging.exception("Callback enhance failed")
        await callback.message.answer(f"Ошибка улучшения: {e}")
    finally:
        await callback.answer()

@dp.callback_query(F.data == VIDEO_CB)
async def cb_video_last(callback: CallbackQuery):
    try:
        user_id = callback.from_user.id
        if not FAL_KEY:
            await callback.message.answer("FAL_KEY не задан — видео отключено.")
            await callback.answer()
            return
        src = last_images.get(user_id)
        if not src:
            await callback.message.answer("Не нашёл последнее фото. Пришли фото ещё раз.")
            await callback.answer()
            return
        await callback.answer("Генерирую видео...")
        prompt = DEFAULT_VIDEO_PROMPT
        video_bytes = await run_with_thinking(callback.message.bot, callback.message.chat.id,
                                              pika_image_bytes_to_video(src, prompt))
        path = _save_bytes_to_tmp(f"pika_{int(time.time())}.mp4", video_bytes)
        await callback.message.answer_video(FSInputFile(path), caption="Видео готово ✅")
    except Exception as e:
        logging.exception("Video generation failed")
        await callback.message.answer(f"Ошибка генерации видео: {e}")
    finally:
        await callback.answer()

@dp.message(F.document)
async def handle_document(message: Message, bot: Bot):
    await _react_ok(message)
    doc = message.document
    filename = doc.file_name or "file"
    ext = _ext(filename)
    mime = doc.mime_type or ""

    try:
        file_bytes, _ = await _download_telegram_file(bot, doc.file_id)
    except Exception as e:
        await message.answer(f"Не удалось загрузить файл: {e}")
        return

    if len(file_bytes) > MAX_FILE_BYTES:
        await message.answer("Файл слишком большой. Пришли поменьше (до ~12MB).")
        return

    last_files[message.from_user.id] = LastFile(filename=filename, ext=ext, mime=mime, data=file_bytes)

    caption = (message.caption or "").strip()
    if caption.lower().startswith("edit:"):
        instructions = caption[5:].strip()
        message.text = f"/edit {instructions}"
        await edit_last_file(message)
        return  # важно: не продолжаем обработку

    # image-as-document
    if ext in {".png", ".jpg", ".jpeg", ".webp"} or mime.startswith("image/"):
        last_images[message.from_user.id] = file_bytes
        await message.answer(
            f"Изображение «{filename}» получил ✅\n"
            "Анализа изображений нет, но могу ✨ улучшить (кнопка/ /enhance) или 🎥 видео (/img2vid)."
        )
        await message.answer("Что сделать с изображением?", reply_markup=_image_action_keyboard())
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
        await message.answer(f"Не смог прочитать файл {filename}. Ошибка: {e}")
        return

    extracted = _safe_truncate(extracted)
    user_hint = caption if caption else "Разбери файл: что в нём важного, ошибки/риски, и что делать дальше."
    system = (
        "Отвечай по делу. Если это таблица — дай выводы и рекомендации. "
        "Если это документ — резюме и конкретные улучшения."
    )
    user = (
        f"Файл: {filename}\n"
        f"Задача: {user_hint}\n\n"
        f"Содержимое:\n-----\n{extracted}\n-----\n"
    )
    try:
        answer = await run_with_thinking(message.bot, message.chat.id, _ask_llm_text(system, user))
    except Exception as e:
        await message.answer(f"LLM error: {e}")
        return

    reference.response = answer
    # FIX: split long message
    for part in split_message(answer):
        await message.answer(part)
    await message.answer("Файл запомнил. Если нужно изменить — напиши /edit <что поменять>.")

@dp.message(F.text)
async def chat(message: Message, bot: Bot):
    try:
        await _react_ok(message)
        user_text = (message.text or "").strip()
        if not user_text or user_text.startswith("/"):
            return

        # If user wrote after sending photo: tell about vision limitation
        if last_images.get(message.from_user.id) and re.search(r"\b(что|кто|где|почему|как|какой|сколько)\b|\?", user_text.lower()):
            await message.answer(
                "Если это вопрос к фото — в этой версии анализа изображений нет.\n"
                "Но могу улучшить фото (/enhance) или сделать видео (/img2vid)."
            )
            return

        # Optional autosearch: if SERPER is configured and user asks for sources
        search_block = ""
        if SERPER_API_KEY and any(k in user_text.lower() for k in ["найди", "источник", "ссылка", "пруф", "актуал", "сейчас", "новост"]):
            try:
                results = await serper_search(user_text, num=5)
                if results:
                    search_block = "\n\nРЕЗУЛЬТАТЫ ПОИСКА:\n" + format_results_for_prompt(results)
            except Exception:
                logging.exception("Autosearch failed")

        system = "Отвечай по делу, человеческим языком. Если есть результаты поиска — опирайся на них и добавь 2–5 ссылок."
        user = user_text + search_block

        answer = await run_with_thinking(message.bot, message.chat.id, _ask_llm_text(system, user))
        reference.response = answer
        # FIX: split long message
        for part in split_message(answer):
            await message.answer(part)
    except Exception as e:
        logging.exception("Unhandled error in chat")
        await message.answer(f"Произошла внутренняя ошибка: {e}")

# ---------- entrypoint ----------
async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())