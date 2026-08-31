import asyncio
import base64
import json
import logging
import math
import mimetypes
import os
import random
import re
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from google import genai
from google.genai import types
import httpx
from openai import OpenAI

# ============================================================
# APP & LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("designmanufaktur")

app = FastAPI(title="Designmanufaktur Super AI Agent + Civil Calculator")

# Global HTTP Client untuk Reuse Connection (Hemat Handshake & Latensi)
http_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup_event():
    global http_client
    http_client = httpx.AsyncClient(timeout=120)


@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()


# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
GEMINI_VIDEO_MODEL = os.getenv(
    "GEMINI_VIDEO_MODEL", "veo-3.1-fast-generate-preview"
)

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_FREE_MODEL = os.getenv("OPENROUTER_FREE_MODEL", "openrouter/free")

NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-flash-0731")
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
NVIDIA_MAX_OUTPUT_TOKENS = int(os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "2048"))

OPENROUTER_BACKUP_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

POLLINATIONS_KEY = os.getenv("POLLINATIONS_API_KEY", "")
POLLINATIONS_ENABLED = (
    os.getenv("POLLINATIONS_ENABLED", "false").lower() == "true"
)
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux")
POLLINATIONS_BASE_URL = "https://gen.pollinations.ai"

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell"
)
CLOUDFLARE_ENABLED = bool(CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN)

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM = """
Kamu adalah Designmanufaktur Super AI Agent.
Kamu adalah asisten AI praktis untuk fabrikasi, teknik, konstruksi ringan, dan civil calculator.
Gunakan Bahasa Indonesia yang lugas, tidak bertele-tele, dan nyaman dibaca di HP.
"""

# ============================================================
# AI CLIENTS
# ============================================================

gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
openrouter = (
    OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    if OPENROUTER_KEY
    else None
)
nvidia = (
    OpenAI(api_key=NVIDIA_KEY, base_url=NVIDIA_BASE_URL) if NVIDIA_KEY else None
)

# ============================================================
# MEMORY & ROUTING MODE
# ============================================================

memory = {}
MAX_MEMORY = 15

AI_MODE_AUTO = "auto"
NVIDIA_STYLE_TASK_MAP = {
    "nvidia_fast": "general",
    "nvidia_coding": "coding",
    "nvidia_technical": "technical",
    "nvidia_reasoning": "reasoning",
}

AI_MODE_CHOICES = {
    "auto": "auto",
    "nvidia_fast": "nvidia_fast",
    "nvidia_coding": "nvidia_coding",
    "nvidia_technical": "nvidia_technical",
    "nvidia_reasoning": "nvidia_reasoning",
    "gemini": "gemini",
    "openrouter": "openrouter",
}

AI_MODE_LABELS = {
    "auto": "🔁 Otomatis (Smart Router + fallback)",
    "nvidia_fast": "⚡ NVIDIA Fast",
    "nvidia_coding": "💻 NVIDIA Coding",
    "nvidia_technical": "🔧 NVIDIA Technical",
    "nvidia_reasoning": "🧠 NVIDIA Reasoning",
    "gemini": "👁️ Gemini",
    "openrouter": "🌐 OpenRouter FREE",
}

user_ai_mode = {}


def get_ai_mode(uid):
  return user_ai_mode.get(str(uid), AI_MODE_AUTO)


def set_ai_mode(uid, mode):
  user_ai_mode[str(uid)] = mode


def build_ai_mode_keyboard():
  def btn(mode):
    return {"text": AI_MODE_LABELS[mode], "callback_data": f"aimode:{mode}"}

  return {
      "inline_keyboard": [
          [btn("auto")],
          [btn("nvidia_fast"), btn("nvidia_coding")],
          [btn("nvidia_technical"), btn("nvidia_reasoning")],
          [btn("gemini"), btn("openrouter")],
      ]
  }


# ============================================================
# PERSISTENT MEMORY (GITHUB)
# ============================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Dinal7297/designmanufaktur-memory")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_MEMORY_DIR = "memory"
GITHUB_API = "https://api.github.com"

WEBSITE_GITHUB_TOKEN = os.getenv("WEBSITE_GITHUB_TOKEN", GITHUB_TOKEN)
WEBSITE_GITHUB_REPO = os.getenv(
    "WEBSITE_GITHUB_REPO", "Dinal7297/Design-Manufaktur-"
)
WEBSITE_GITHUB_BRANCH = os.getenv("WEBSITE_GITHUB_BRANCH", "main")
WEBSITE_DATA_PATH = "data/pekerjaan.json"

MAX_CONTEXT_TURNS = 6
MAX_CONTEXT_CHARS_PER_ITEM = 1000
NVIDIA_MAX_CONTEXT_TURNS = 4
NVIDIA_MAX_CONTEXT_CHARS_PER_ITEM = 600
OPENROUTER_MAX_OUTPUT_TOKENS = 2048


def history(uid):
  return memory.setdefault(uid, [])


def remember(uid, role, content):
  history(uid).append({"role": role, "content": content})
  memory[uid] = history(uid)[-MAX_MEMORY:]


def _trim_history_for_context(
    uid,
    max_turns=MAX_CONTEXT_TURNS,
    max_chars_per_item=MAX_CONTEXT_CHARS_PER_ITEM,
):
  items = history(uid)[-max_turns:]
  trimmed = []
  for m in items:
    content = m.get("content", "") or ""
    if len(content) > max_chars_per_item:
      content = content[:max_chars_per_item] + "\n...(dipotong)"
    trimmed.append({"role": m.get("role", "user"), "content": content})
  return trimmed


def _memory_path(uid):
  return f"{GITHUB_MEMORY_DIR}/{str(uid)}.json"


async def load_persistent_memory(uid):
  uid = str(uid)
  if not GITHUB_TOKEN or not GITHUB_REPO:
    memory.setdefault(uid, [])
    return
  url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{_memory_path(uid)}"
  headers = {
      "Authorization": f"Bearer {GITHUB_TOKEN}",
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
  }
  try:
    client = http_client or httpx.AsyncClient(timeout=10)
    response = await client.get(
        url, headers=headers, params={"ref": GITHUB_BRANCH}
    )
    if response.status_code == 404:
      memory[uid] = []
      return
    response.raise_for_status()
    encoded = response.json().get("content", "")
    if not encoded:
      memory[uid] = []
      return
    raw = base64.b64decode(encoded.replace("\n", "")).decode("utf-8")
    saved = json.loads(raw)
    if isinstance(saved, dict):
      saved_mode = saved.get("ai_mode")
      if saved_mode in AI_MODE_LABELS:
        user_ai_mode[uid] = saved_mode
      saved = saved.get("memory", [])
    memory[uid] = saved[-MAX_MEMORY:] if isinstance(saved, list) else []
  except Exception as e:
    log.warning("LOAD MEMORY FAILED | uid=%s | %s", uid, str(e)[:150])
    memory.setdefault(uid, [])


async def save_persistent_memory(uid):
  uid = str(uid)
  if not GITHUB_TOKEN or not GITHUB_REPO:
    return
  raw = json.dumps(
      {
          "user_id": uid,
          "memory": history(uid)[-MAX_MEMORY:],
          "ai_mode": get_ai_mode(uid),
          "updated_at": int(time.time()),
      },
      ensure_ascii=False,
  )
  encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
  url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{_memory_path(uid)}"
  headers = {
      "Authorization": f"Bearer {GITHUB_TOKEN}",
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
  }
  try:
    client = http_client or httpx.AsyncClient(timeout=10)
    current = await client.get(
        url, headers=headers, params={"ref": GITHUB_BRANCH}
    )
    body = {
        "message": f"update memory {uid}",
        "content": encoded,
        "branch": GITHUB_BRANCH,
    }
    if current.status_code == 200:
      body["sha"] = current.json().get("sha")
    await client.put(url, headers=headers, json=body)
  except Exception as e:
    log.warning("SAVE MEMORY FAILED | uid=%s | %s", uid, str(e)[:150])


# ============================================================
# TELEGRAM API HELPERS
# ============================================================


async def tg(method, data):
  if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN belum diatur.")
  client = http_client or httpx.AsyncClient(timeout=30)
  r = await client.post(
      f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}", json=data
  )
  r.raise_for_status()
  result = r.json()
  if not result.get("ok"):
    raise RuntimeError(str(result))
  return result


async def tg_file(file_id):
  result = await tg("getFile", {"file_id": file_id})
  path = result["result"]["file_path"]
  client = http_client or httpx.AsyncClient(timeout=60)
  r = await client.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}")
  r.raise_for_status()
  return r.content, path


# ============================================================
# SMART CHAT ROUTER & AI CALLS
# ============================================================


def build_messages(
    uid,
    text,
    task,
    max_turns=MAX_CONTEXT_TURNS,
    max_chars_per_item=MAX_CONTEXT_CHARS_PER_ITEM,
):
  return (
      [{"role": "system", "content": SYSTEM}]
      + _trim_history_for_context(uid, max_turns, max_chars_per_item)
      + [{"role": "user", "content": text}]
  )


def call_nvidia(uid, text, task, model=None):
  if not nvidia:
    raise RuntimeError("NVIDIA_API_KEY belum tersedia.")
  selected = model or NVIDIA_MODEL
  r = nvidia.chat.completions.create(
      model=selected,
      messages=build_messages(
          uid,
          text,
          task,
          NVIDIA_MAX_CONTEXT_TURNS,
          NVIDIA_MAX_CONTEXT_CHARS_PER_ITEM,
      ),
      max_tokens=NVIDIA_MAX_OUTPUT_TOKENS,
  )
  answer = r.choices[0].message.content or ""
  if not answer.strip():
    raise RuntimeError("NVIDIA kosong.")
  return answer, selected


def call_openrouter(uid, text, task, model=None):
  if not openrouter:
    raise RuntimeError("OPENROUTER_API_KEY belum tersedia.")
  selected = model or OPENROUTER_FREE_MODEL
  r = openrouter.chat.completions.create(
      model=selected,
      messages=build_messages(uid, text, task),
      max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
  )
  answer = r.choices[0].message.content or ""
  if not answer.strip():
    raise RuntimeError("OpenRouter kosong.")
  return answer, getattr(r, "model", selected)


def call_gemini(uid, text, task):
  if not gemini:
    raise RuntimeError("GEMINI_API_KEY belum tersedia.")
  prompt = SYSTEM + "\n\n"
  for m in _trim_history_for_context(uid):
    prompt += f"{m['role']}: {m['content']}\n"
  prompt += f"user: {text}"
  r = gemini.models.generate_content(
      model=GEMINI_CHAT_MODEL, contents=prompt
  )
  answer = r.text or ""
  if not answer.strip():
    raise RuntimeError("Gemini kosong.")
  return answer, GEMINI_CHAT_MODEL


def chat_router(uid, text, ai_mode="auto"):
  start_time = time.time()
  task = "general"

  providers = [
      ("⚡ NVIDIA", lambda: call_nvidia(uid, text, task)),
      ("🌐 OpenRouter", lambda: call_openrouter(uid, text, task)),
      ("👁️ Gemini", lambda: call_gemini(uid, text, task)),
  ]

  if ai_mode in NVIDIA_STYLE_TASK_MAP:
    providers = [(
        AI_MODE_LABELS[ai_mode],
        lambda: call_nvidia(uid, text, NVIDIA_STYLE_TASK_MAP[ai_mode]),
    )]
  elif ai_mode == "gemini":
    providers = [("👁️ Gemini", lambda: call_gemini(uid, text, task))]
  elif ai_mode == "openrouter":
    providers = [("🌐 OpenRouter", lambda: call_openrouter(uid, text, task))]

  attempts = []
  for name, fn in providers:
    try:
      answer, model = fn()
      elapsed = round(time.time() - start_time, 2)
      attempts.append({"provider": name, "model": model, "status": "ok"})
      return answer, name, model, task, attempts, elapsed
    except Exception as e:
      attempts.append(
          {"provider": name, "status": "failed", "error": str(e)[:100]}
      )

  raise RuntimeError("Semua provider AI gagal.")


# ============================================================
# IMAGE GENERATION (ASYNC SAFE)
# ============================================================


def generate_image_sync(prompt):
  if CLOUDFLARE_ENABLED:
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    with httpx.Client(timeout=45) as client:
      r = client.post(
          url,
          headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
          json={"prompt": prompt},
      )
      if r.status_code == 200:
        return r.content, "Cloudflare FLUX"

  if POLLINATIONS_ENABLED:
    from urllib.parse import quote

    url = f"{POLLINATIONS_BASE_URL}/image/{quote(prompt, safe='')}?model={POLLINATIONS_IMAGE_MODEL}"
    with httpx.Client(timeout=45) as client:
      r = client.get(
          url, headers={"Authorization": f"Bearer {POLLINATIONS_KEY}"}
      )
      if r.status_code == 200:
        return r.content, "Pollinations"

  raise RuntimeError("Generator gambar tidak tersedia.")


# ============================================================
# MAIN TELEGRAM UPDATES HANDLER
# ============================================================


async def handle(update):
  callback_query = update.get("callback_query")
  if callback_query:
    cq_id = callback_query.get("id")
    chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
    uid = str(callback_query.get("from", {}).get("id", chat_id))
    data = callback_query.get("data", "")

    if data.startswith("aimode:"):
      mode = data.split(":", 1)[1]
      await load_persistent_memory(uid)
      set_ai_mode(uid, mode)
      await save_persistent_memory(uid)
      await tg("answerCallbackQuery", {"callback_query_id": cq_id})
      await tg(
          "sendMessage",
          {"chat_id": chat_id, "text": f"✅ Mode diubah ke: {mode}"},
      )
    return

  message = update.get("message")
  if not message:
    return

  chat_id = message.get("chat", {}).get("id")
  uid = str(message.get("from", {}).get("id", chat_id))
  text = message.get("text", "")

  if text.startswith("/start"):
    await tg(
        "sendMessage",
        {"chat_id": chat_id, "text": "🤖 Bot Aktif dan Siap Digunakan!"},
    )
    return

  if text.startswith("/ai"):
    await load_persistent_memory(uid)
    arg = text.replace("/ai", "").strip().lower()
    if arg in AI_MODE_CHOICES:
      set_ai_mode(uid, AI_MODE_CHOICES[arg])
      await save_persistent_memory(uid)
      await tg(
          "sendMessage",
          {"chat_id": chat_id, "text": f"✅ Mode AI diatur ke: {arg}"},
      )
    else:
      await tg(
          "sendMessage",
          {
              "chat_id": chat_id,
              "text": "Pilih mode:",
              "reply_markup": build_ai_mode_keyboard(),
          },
      )
    return

  if text.startswith("/gambar"):
    prompt = text.replace("/gambar", "").strip()
    if not prompt:
      await tg(
          "sendMessage",
          {"chat_id": chat_id, "text": "Masukkan prompt gambar."},
      )
      return
    try:
      img_bytes, provider = await asyncio.to_thread(
          generate_image_sync, prompt
      )
      client = http_client or httpx.AsyncClient(timeout=60)
      await client.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
          data={"chat_id": str(chat_id), "caption": f"✅ Gambar by {provider}"},
          files={"photo": ("image.jpg", img_bytes, "image/jpeg")},
      )
    except Exception as e:
      await tg(
          "sendMessage",
          {"chat_id": chat_id, "text": f"❌ Gagal buat gambar: {e}"},
      )
    return

  if text:
    await load_persistent_memory(uid)
    ai_mode = get_ai_mode(uid)
    try:
      answer, provider, model, task, attempts, elapsed = await asyncio.to_thread(
          chat_router, uid, text, ai_mode
      )
      remember(uid, "user", text)
      remember(uid, "assistant", answer)
      await save_persistent_memory(uid)

      res_text = f"{answer}\n\n🤖 {provider} ({model}) • ⏱️ {elapsed}s"
      await tg("sendMessage", {"chat_id": chat_id, "text": res_text})
    except Exception as e:
      await tg(
          "sendMessage",
          {"chat_id": chat_id, "text": f"❌ Error: {str(e)[:300]}"},
      )


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/")
async def root():
  return {"ok": True, "status": "running"}


@app.post("/api/webhook")
@app.post("/")
async def webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
  if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
    raise HTTPException(status_code=403, detail="Invalid secret")
  update = await request.json()
  asyncio.create_task(handle(update))  # Non-blocking processing
  return {"ok": True}
