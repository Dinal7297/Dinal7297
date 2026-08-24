import asyncio
import base64
import io
import logging
import mimetypes
import os
import re
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from google import genai
from google.genai import types
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("designmanufaktur")
app = FastAPI(title="Designmanufaktur Super AI Agent")

# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

# Gemini: KEEP the Gemini setup that already works in your bot.
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"
)
GEMINI_VIDEO_MODEL = os.getenv(
    "GEMINI_VIDEO_MODEL", "veo-3.1-fast-generate-preview"
)

# OpenRouter FREE ONLY
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_FREE_MODEL = "openrouter/free"

# Groq FREE-TIER fallback.
# IMPORTANT: this is Groq, not xAI/Grok.
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_FAST_MODEL = os.getenv(
    "GROQ_FAST_MODEL", "openai/gpt-oss-20b"
)
GROQ_REASONING_MODEL = os.getenv(
    "GROQ_REASONING_MODEL", "openai/gpt-oss-20b"
)
GROQ_CODING_MODEL = os.getenv(
    "GROQ_CODING_MODEL", "qwen/qwen3-32b"
)

# Optional Pollinations image generation
POLLINATIONS_KEY = os.getenv("POLLINATIONS_API_KEY", "")
POLLINATIONS_ENABLED = os.getenv(
    "POLLINATIONS_ENABLED", "false"
).lower() == "true"
POLLINATIONS_IMAGE_MODEL = os.getenv(
    "POLLINATIONS_IMAGE_MODEL", "flux"
)
POLLINATIONS_BASE_URL = "https://gen.pollinations.ai"

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM = """Kamu adalah Designmanufaktur Super AI Agent.

Kamu membantu pekerjaan:
- manufaktur
- bengkel las
- fabrikasi
- tenda
- pagar
- konstruksi
- desain produk custom
- engineering
- coding/programming
- analisis
- bisnis dan konten

Jawab dalam Bahasa Indonesia kecuali pengguna meminta bahasa lain.
Jawab jelas, akurat, praktis, dan tidak bertele-tele.
Gunakan kemampuan model secara maksimal untuk jenis tugas yang diberikan.
Jika data tidak tersedia, katakan bahwa data tersebut belum tersedia.
Jangan mengarang ukuran, harga, spesifikasi, atau fakta.
Untuk perhitungan, tampilkan asumsi dan langkah penting.
Jangan pernah menampilkan API key, token, password, atau rahasia sistem.

PRINSIP ROUTER:
- Sistem memilih AI berdasarkan jenis tugas, bukan rotasi sederhana.
- Prioritas utama adalah provider/model GRATIS.
- Jangan sengaja memakai model berbayar.
- Jika provider gratis gagal, terkena rate limit, timeout, atau tidak tersedia,
  otomatis coba provider gratis berikutnya.
"""

gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

openrouter = (
    OpenAI(
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    if OPENROUTER_KEY
    else None
)

groq = (
    OpenAI(
        api_key=GROQ_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    if GROQ_KEY
    else None
)

# ============================================================
# EPHEMERAL CHAT MEMORY
# NOTE: GitHub persistent memory can be added later without
# changing the router architecture.
# ============================================================

memory = {}
MAX_MEMORY = 20


def history(uid):
    return memory.setdefault(uid, [])


def remember(uid, role, content):
    history(uid).append({"role": role, "content": content})
    memory[uid] = history(uid)[-MAX_MEMORY:]


def build_messages(uid, text, task):
    task_hint = {
        "coding": "Tugas coding: prioritaskan ketepatan kode, debugging, dan solusi yang dapat dijalankan.",
        "reasoning": "Tugas reasoning: analisis langkah demi langkah secara ringkas dan verifikasi kesimpulan.",
        "technical": "Tugas teknik/manufaktur: prioritaskan engineering praktis, material, fabrikasi, ukuran, dan keselamatan.",
        "math": "Tugas matematika: hitung dengan teliti dan tunjukkan rumus/asumsi.",
        "creative": "Tugas kreatif: buat hasil yang praktis dan sesuai tujuan pengguna.",
        "general": "Tugas umum: berikan jawaban langsung dan berguna.",
    }.get(task, "")

    return [
        {
            "role": "system",
            "content": SYSTEM + "\n\nKATEGORI TUGAS:\n" + task_hint,
        }
    ] + history(uid) + [{"role": "user", "content": text}]


# ============================================================
# TASK CLASSIFIER
# ============================================================

def classify_task(text):
    t = (text or "").lower()

    coding = [
        "python", "javascript", "typescript", "php", "html", "css",
        "sql", "api", "coding", "kode", "program", "programming",
        "bug", "error", "debug", "github", "vercel", "function",
        "class ", "import ", "async ", "def ", "javascript"
    ]
    reasoning = [
        "analisis", "analisa", "kenapa", "mengapa", "bandingkan",
        "perbandingan", "strategi", "logika", "alasan", "evaluasi",
        "pecahkan", "solusi terbaik", "reasoning"
    ]
    technical = [
        "tenda", "rangka", "hollow", "pipa", "baja", "las",
        "fabrikasi", "manufaktur", "produksi", "material", "plat",
        "besi", "aluminium", "konstruksi", "ukuran", "dimensi",
        "pagar", "kanopi", "bengkel", "welding", "engineering"
    ]
    math = [
        "hitung", "perhitungan", "berapa", "rumus", "luas",
        "volume", "persentase", "matematika", "kg", "meter",
        "mm", "cm", "m2", "m²"
    ]
    creative = [
        "caption", "iklan", "promosi", "slogan", "desain",
        "buatkan gambar", "ide konten", "copywriting"
    ]

    if any(x in t for x in coding):
        return "coding"
    if any(x in t for x in math):
        return "math"
    if any(x in t for x in technical):
        return "technical"
    if any(x in t for x in reasoning):
        return "reasoning"
    if any(x in t for x in creative):
        return "creative"
    return "general"


# ============================================================
# PROVIDER CALLS
# ============================================================

def call_openrouter(uid, text, task):
    if not openrouter:
        raise RuntimeError("OPENROUTER_API_KEY belum tersedia.")

    # IMPORTANT:
    # openrouter/free is guaranteed to use free models only.
    # It selects a suitable free model based on request capabilities.
    r = openrouter.chat.completions.create(
        model=OPENROUTER_FREE_MODEL,
        messages=build_messages(uid, text, task),
        max_tokens=4096,
        extra_headers={
            "HTTP-Referer": "https://designmanufaktur.vercel.app",
            "X-Title": "Designmanufaktur Super AI Agent",
        },
    )

    answer = r.choices[0].message.content or ""
    if not answer.strip():
        raise RuntimeError("OpenRouter Free mengembalikan jawaban kosong.")

    selected_model = getattr(r, "model", None) or "openrouter/free"
    return answer, selected_model


def call_gemini(uid, text, task):
    if not gemini:
        raise RuntimeError("GEMINI_API_KEY belum tersedia.")

    task_hint = {
        "coding": "Berikan kode yang dapat dijalankan dan jelaskan perubahan penting.",
        "reasoning": "Analisis masalah secara teliti sebelum memberi kesimpulan.",
        "technical": "Gunakan pertimbangan teknik/manufaktur yang praktis.",
        "math": "Hitung secara teliti dan tunjukkan asumsi.",
        "creative": "Buat hasil kreatif yang siap digunakan.",
        "general": "Jawab langsung dan jelas.",
    }.get(task, "")

    prompt = SYSTEM + "\n\n" + task_hint + "\n\n"
    for m in history(uid):
        prompt += f"{m['role']}: {m['content']}\n"
    prompt += f"user: {text}"

    r = gemini.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
    )
    answer = r.text or ""
    if not answer.strip():
        raise RuntimeError("Gemini mengembalikan jawaban kosong.")

    return answer, GEMINI_CHAT_MODEL


def call_groq(uid, text, task):
    if not groq:
        raise RuntimeError("GROQ_API_KEY belum tersedia.")

    if task == "coding":
        model = GROQ_CODING_MODEL
    elif task in ("reasoning", "math"):
        model = GROQ_REASONING_MODEL
    else:
        model = GROQ_FAST_MODEL

    r = groq.chat.completions.create(
        model=model,
        messages=build_messages(uid, text, task),
        max_tokens=4096,
    )

    answer = r.choices[0].message.content or ""
    if not answer.strip():
        raise RuntimeError("Groq mengembalikan jawaban kosong.")

    return answer, model


# ============================================================
# SMART CHAT ROUTER
# ============================================================

def chat_router(uid, text):
    task = classify_task(text)
    log.info("TASK=%s | text=%s", task, text[:120])

    errors = []

    # --------------------------------------------------------
    # TASK-BASED PRIORITY
    #
    # technical -> Gemini first because this is the proven
    # Designmanufaktur specialist.
    #
    # coding/reasoning/math/general/creative -> OpenRouter Free
    # first, because FREE is the main priority.
    # --------------------------------------------------------

    if task == "technical":
        providers = [
            ("Gemini", lambda: call_gemini(uid, text, task)),
            ("OpenRouter Free", lambda: call_openrouter(uid, text, task)),
            ("Groq Free Tier", lambda: call_groq(uid, text, task)),
        ]
    elif task in ("coding", "reasoning", "math"):
        providers = [
            ("OpenRouter Free", lambda: call_openrouter(uid, text, task)),
            ("Gemini", lambda: call_gemini(uid, text, task)),
            ("Groq Free Tier", lambda: call_groq(uid, text, task)),
        ]
    else:
        providers = [
            ("OpenRouter Free", lambda: call_openrouter(uid, text, task)),
            ("Gemini", lambda: call_gemini(uid, text, task)),
            ("Groq Free Tier", lambda: call_groq(uid, text, task)),
        ]

    for provider_name, fn in providers:
        try:
            answer, model = fn()
            log.info(
                "CHAT SUCCESS | task=%s | provider=%s | model=%s",
                task,
                provider_name,
                model,
            )
            return answer, provider_name, model, task
        except Exception as e:
            log.exception("%s failed", provider_name)
            errors.append(
                f"{provider_name}: {str(e)[:220]}"
            )

    raise RuntimeError(
        "Semua provider GRATIS gagal: " + " | ".join(errors)
    )


# ============================================================
# TELEGRAM
# ============================================================

async def tg(method, data):
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN belum diatur.")

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
            json=data,
        )
        r.raise_for_status()
        result = r.json()

    if not result.get("ok"):
        raise RuntimeError(str(result))

    return result


async def tg_file(file_id):
    result = await tg("getFile", {"file_id": file_id})
    path = result["result"]["file_path"]

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(
            f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}"
        )
        r.raise_for_status()

    return r.content, path


async def send_text(chat_id, text):
    text = text or "Tidak ada jawaban."

    for i in range(0, len(text), 3900):
        await tg(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text[i:i + 3900],
            },
        )


async def send_photo(chat_id, data):
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={"chat_id": str(chat_id)},
            files={
                "photo": (
                    "image.png",
                    data,
                    "image/png",
                )
            },
        )
        r.raise_for_status()


async def send_video(chat_id, data):
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
            data={"chat_id": str(chat_id)},
            files={
                "video": (
                    "video.mp4",
                    data,
                    "video/mp4",
                )
            },
        )
        r.raise_for_status()


# ============================================================
# GEMINI VISION
# ============================================================

def analyze_image(data, mime, prompt):
    if not gemini:
        raise RuntimeError("Gemini belum dikonfigurasi.")

    errors = []

    # PRIMARY: the Gemini model that is already proven to work
    # in the user's bot. We use the current chat model because
    # Gemini Flash supports image input + text output.
    try:
        r = gemini.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=data,
                    mime_type=mime,
                ),
                SYSTEM + "\n\n" + prompt,
            ],
        )

        if r.text:
            return r.text, GEMINI_CHAT_MODEL

    except Exception as e:
        errors.append(f"Gemini Vision: {str(e)[:220]}")

    # FREE MULTI-MODEL VISION FALLBACK
    if openrouter:
        try:
            b64 = base64.b64encode(data).decode()
            content = [
                {
                    "type": "text",
                    "text": SYSTEM + "\n\n" + prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}"
                    },
                },
            ]

            r = openrouter.chat.completions.create(
                model=OPENROUTER_FREE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=4096,
            )

            answer = r.choices[0].message.content or ""
            if answer.strip():
                return answer, getattr(
                    r,
                    "model",
                    "openrouter/free",
                )

        except Exception as e:
            errors.append(
                f"OpenRouter Free Vision: {str(e)[:220]}"
            )

    raise RuntimeError(
        "Semua provider vision gratis gagal: "
        + " | ".join(errors)
    )


# ============================================================
# GEMINI VIDEO ANALYSIS
# ============================================================

def analyze_video(data, mime, prompt):
    if not gemini:
        raise RuntimeError("Gemini diperlukan untuk analisis video.")

    uploaded = gemini.files.upload(
        file=types.Part.from_bytes(
            data=data,
            mime_type=mime,
        )
    )

    for _ in range(60):
        f = gemini.files.get(name=uploaded.name)
        state = getattr(
            getattr(f, "state", None),
            "name",
            "",
        )

        if state == "ACTIVE":
            uploaded = f
            break

        if state == "FAILED":
            raise RuntimeError(
                "Gemini gagal memproses video."
            )

        time.sleep(2)
    else:
        raise RuntimeError(
            "Video belum siap diproses."
        )

    result = gemini.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=[
            uploaded,
            SYSTEM + "\n\n" + prompt,
        ],
    )

    return result.text or ""


# ============================================================
# IMAGE GENERATION
# FREE ONLY
# ============================================================

def pollinations_image(prompt):
    if not POLLINATIONS_ENABLED:
        raise RuntimeError("Pollinations tidak diaktifkan.")

    if not POLLINATIONS_KEY:
        raise RuntimeError(
            "POLLINATIONS_API_KEY belum tersedia."
        )

    from urllib.parse import quote

    url = (
        f"{POLLINATIONS_BASE_URL}/image/"
        f"{quote(prompt, safe='')}"
        f"?model={quote(POLLINATIONS_IMAGE_MODEL)}"
        f"&width=1024&height=1024"
    )

    with httpx.Client(timeout=300) as client:
        r = client.get(
            url,
            headers={
                "Authorization": (
                    f"Bearer {POLLINATIONS_KEY}"
                ),
                "Accept": "image/png,image/jpeg,*/*",
            },
        )

        if r.status_code >= 400:
            raise RuntimeError(
                f"Pollinations HTTP {r.status_code}: "
                f"{r.text[:400]}"
            )

        if not r.content:
            raise RuntimeError(
                "Pollinations mengembalikan data kosong."
            )

        return r.content


def generate_image(prompt):
    # FREE ONLY: Pollinations first.
    # Gemini image generation is intentionally NOT used here
    # because this project must avoid paid fallbacks.
    if POLLINATIONS_ENABLED:
        return pollinations_image(prompt), "Pollinations"

    raise RuntimeError(
        "Generate gambar GRATIS belum tersedia. "
        "Aktifkan POLLINATIONS_ENABLED=true dan "
        "POLLINATIONS_API_KEY."
    )


# ============================================================
# COMMANDS
# ============================================================

def command_arg(text):
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


async def handle(update):
    message = update.get("message")

    if not message:
        return

    chat_id = message.get("chat", {}).get("id")
    uid = str(
        message.get("from", {}).get(
            "id",
            chat_id,
        )
    )

    text = message.get("text", "") or ""
    caption = message.get("caption", "") or ""

    # --------------------------------------------------------
    # /start
    # --------------------------------------------------------

    if text.startswith("/start"):
        await send_text(
            chat_id,
            """🤖 Designmanufaktur Super AI Agent aktif.

🧠 Smart Multi-AI Router
💰 FREE-FIRST
🖼️ Gemini Vision
🎥 Gemini Video Analysis
🎨 Free Image Generation

Chat biasa → AI dipilih sesuai jenis tugas.
Coding → model gratis yang cocok.
Reasoning → model gratis yang cocok.
Teknik/manufaktur → Gemini specialist.
Foto → Gemini Vision.
Jika provider gagal → otomatis fallback.

/model → status AI
/reset → hapus memory sesi
/gambar <prompt> → generate gambar gratis
/video <prompt> → analisis/generate video sesuai kemampuan gratis""",
        )
        return

    # --------------------------------------------------------
    # /reset
    # --------------------------------------------------------

    if text.startswith("/reset"):
        memory.pop(uid, None)
        await send_text(
            chat_id,
            "✅ Memory sesi dihapus.",
        )
        return

    # --------------------------------------------------------
    # /model
    # --------------------------------------------------------

    if text.startswith("/model"):
        await send_text(
            chat_id,
            f"""STATUS SUPER AI AGENT

Gemini: {'✅' if gemini else '❌'}
OpenRouter FREE: {'✅' if openrouter else '❌'}
Groq FREE-TIER: {'✅' if groq else '❌'}

Gemini chat:
{GEMINI_CHAT_MODEL}

OpenRouter:
{OPENROUTER_FREE_MODEL}

Groq coding:
{GROQ_CODING_MODEL}

Groq reasoning:
{GROQ_REASONING_MODEL}

Groq fast:
{GROQ_FAST_MODEL}

Gemini Vision:
{GEMINI_CHAT_MODEL}

Image generation FREE:
{'Pollinations ✅' if POLLINATIONS_ENABLED and POLLINATIONS_KEY else 'Tidak aktif'}

ROUTING:
Technical/Manufacturing → Gemini → OpenRouter Free → Groq
Coding/Reasoning/Math → OpenRouter Free → Gemini → Groq
General/Creative → OpenRouter Free → Gemini → Groq
Vision → Gemini → OpenRouter Free

PAID MODEL ROUTING: DISABLED""",
        )
        return

    # --------------------------------------------------------
    # /gambar
    # --------------------------------------------------------

    if text.startswith("/gambar"):
        prompt = command_arg(text)

        if not prompt:
            await send_text(
                chat_id,
                "Contoh:\n/gambar pagar minimalis hitam modern",
            )
            return

        await send_text(
            chat_id,
            "🎨 Memilih generator gambar GRATIS...",
        )

        try:
            data, provider = await asyncio.to_thread(
                generate_image,
                prompt,
            )

            await send_photo(
                chat_id,
                data,
            )

            await send_text(
                chat_id,
                f"✅ Gambar dibuat oleh {provider}.",
            )

        except Exception as e:
            log.exception("image generation failed")
            await send_text(
                chat_id,
                "❌ Generate gambar gratis gagal.\n"
                + str(e)[:700],
            )

        return

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if message.get("video"):
        await send_text(
            chat_id,
            "🎥 Sedang menganalisis video...",
        )

        try:
            data, path = await tg_file(
                message["video"]["file_id"]
            )

            if len(data) > 20 * 1024 * 1024:
                await send_text(
                    chat_id,
                    "❌ Video lebih dari 20 MB.",
                )
                return

            mime = (
                "video/quicktime"
                if path.lower().endswith(".mov")
                else "video/mp4"
            )

            answer = await asyncio.to_thread(
                analyze_video,
                data,
                mime,
                caption or (
                    "Analisa video ini secara detail. "
                    "Jelaskan objek, proses, kondisi, "
                    "masalah yang terlihat, dan saran praktis."
                ),
            )

            await send_text(
                chat_id,
                answer,
            )

        except Exception as e:
            log.exception("video analysis failed")
            await send_text(
                chat_id,
                "❌ Analisis video gagal.\n"
                + str(e)[:700],
            )

        return

    # --------------------------------------------------------
    # PHOTO
    # --------------------------------------------------------

    if message.get("photo"):
        await send_text(
            chat_id,
            "🖼️ Gemini Vision sedang menganalisis gambar...",
        )

        try:
            data, path = await tg_file(
                message["photo"][-1]["file_id"]
            )

            mime = (
                mimetypes.guess_type(path)[0]
                or "image/jpeg"
            )

            prompt = caption or (
                "Analisa gambar ini secara detail. "
                "Jika terkait manufaktur, bengkel las, "
                "tenda, pagar, fabrikasi, konstruksi, "
                "atau produk custom, jelaskan objek/komponen "
                "yang terlihat, fungsi, kondisi, kemungkinan "
                "ukuran yang dapat diperkirakan secara visual, "
                "masalah yang terlihat, dan saran praktis. "
                "Jangan mengarang ukuran atau data yang tidak "
                "terlihat pada gambar."
            )

            answer, model = await asyncio.to_thread(
                analyze_image,
                data,
                mime,
                prompt,
            )

            await send_text(
                chat_id,
                answer,
            )

            log.info(
                "VISION SUCCESS | model=%s",
                model,
            )

        except Exception as e:
            log.exception("image analysis failed")
            await send_text(
                chat_id,
                "❌ Analisis gambar gagal.\n"
                + str(e)[:700],
            )

        return

    # --------------------------------------------------------
    # NORMAL CHAT
    # --------------------------------------------------------

    if not text:
        return

    try:
        await tg(
            "sendChatAction",
            {
                "chat_id": chat_id,
                "action": "typing",
            },
        )

        answer, provider, model, task = (
            await asyncio.to_thread(
                chat_router,
                uid,
                text,
            )
        )

        remember(
            uid,
            "user",
            text,
        )

        remember(
            uid,
            "assistant",
            answer,
        )

        await send_text(
            chat_id,
            answer,
        )

        log.info(
            "CHAT DONE | task=%s | provider=%s | model=%s",
            task,
            provider,
            model,
        )

    except Exception as e:
        log.exception("chat failed")

        await send_text(
            chat_id,
            "❌ Semua AI GRATIS gagal untuk request ini.\n"
            + str(e)[:700],
        )


# ============================================================
# WEB
# ============================================================

@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "Designmanufaktur Super AI Agent",
        "free_only": True,
        "providers": {
            "gemini": bool(gemini),
            "openrouter_free": bool(openrouter),
            "groq_free_tier": bool(groq),
        },
        "models": {
            "gemini": GEMINI_CHAT_MODEL,
            "openrouter": OPENROUTER_FREE_MODEL,
            "groq_coding": GROQ_CODING_MODEL,
            "groq_reasoning": GROQ_REASONING_MODEL,
            "groq_fast": GROQ_FAST_MODEL,
        },
    }


@app.get("/api")
async def api_root():
    return await root()


@app.post("/api/webhook")
async def webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    if (
        WEBHOOK_SECRET
        and x_telegram_bot_api_secret_token
        != WEBHOOK_SECRET
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook secret",
        )

    update = await request.json()
    await handle(update)

    return {"ok": True}


@app.post("/")
async def root_post(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    if (
        WEBHOOK_SECRET
        and x_telegram_bot_api_secret_token
        != WEBHOOK_SECRET
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook secret",
        )

    update = await request.json()
    await handle(update)

    return {"ok": True}
