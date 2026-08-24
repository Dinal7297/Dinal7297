import asyncio
import base64
import mimetypes
import os
import re
import time
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from google import genai
from google.genai import types
from openai import OpenAI


# ============================================================
# APP
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("designmanufaktur")

app = FastAPI(
    title="Designmanufaktur Super AI Agent"
)


# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_TOKEN",
    ""
)

WEBHOOK_SECRET = os.getenv(
    "TELEGRAM_WEBHOOK_SECRET",
    ""
)


# ============================================================
# GEMINI
# ============================================================

GEMINI_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

GEMINI_CHAT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

GEMINI_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL",
    "gemini-2.5-flash-image"
)

GEMINI_VIDEO_MODEL = os.getenv(
    "GEMINI_VIDEO_MODEL",
    "veo-3.1-fast-generate-preview"
)


# ============================================================
# OPENROUTER FREE
# ============================================================

OPENROUTER_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
)

OPENROUTER_FREE_MODEL = "openrouter/free"


# ============================================================
# GROQ FREE TIER
# ============================================================

GROQ_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
)

GROQ_FAST_MODEL = os.getenv(
    "GROQ_FAST_MODEL",
    "openai/gpt-oss-20b"
)

GROQ_REASONING_MODEL = os.getenv(
    "GROQ_REASONING_MODEL",
    "openai/gpt-oss-20b"
)

GROQ_CODING_MODEL = os.getenv(
    "GROQ_CODING_MODEL",
    "qwen/qwen3-32b"
)


# ============================================================
# POLLINATIONS IMAGE
# ============================================================

POLLINATIONS_KEY = os.getenv(
    "POLLINATIONS_API_KEY",
    ""
)

POLLINATIONS_ENABLED = os.getenv(
    "POLLINATIONS_ENABLED",
    "false"
).lower() == "true"

POLLINATIONS_IMAGE_MODEL = os.getenv(
    "POLLINATIONS_IMAGE_MODEL",
    "flux"
)

POLLINATIONS_BASE_URL = (
    "https://gen.pollinations.ai"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM = """
Kamu adalah Designmanufaktur Super AI Agent.

Kamu adalah asisten AI praktis untuk pekerjaan:

- bengkel las
- kanopi
- tenda
- pagar
- rangka baja
- hollow
- pipa
- fabrikasi
- manufaktur
- konstruksi ringan
- desain produk custom
- cutting list
- estimasi material
- engineering
- perhitungan teknis
- coding/programming
- bisnis
- konten dan pemasaran

BAHASA:
Jawab dalam Bahasa Indonesia kecuali pengguna meminta bahasa lain.

GAYA JAWABAN:
- langsung ke inti
- praktis
- jelas
- tidak bertele-tele
- gunakan tabel jika membantu
- gunakan satuan yang jelas
- jangan membuat jawaban terlihat rumit tanpa alasan

ATURAN AKURASI:
1. Jangan mengarang ukuran, harga, material, beban, kapasitas,
   atau spesifikasi yang tidak diberikan pengguna.

2. Jika data belum tersedia, nyatakan secara jelas:
   "Data belum ditentukan."

3. Untuk perhitungan:
   - tuliskan asumsi
   - tuliskan rumus penting
   - hitung hasilnya
   - tuliskan hasil akhir
   - gunakan satuan yang konsisten

4. Jika pengguna meminta kebutuhan batang 6 meter:
   - hitung total kebutuhan panjang
   - susun cutting list
   - optimalkan kombinasi potongan
   - hitung sisa/waste
   - jangan hanya membagi total panjang dengan 6
   - perhatikan bahwa satu batang tidak boleh melebihi 6 meter

5. Jika ada potongan yang membutuhkan sambungan:
   jelaskan bahwa sambungan diperlukan dan jangan menganggap
   material menyambung otomatis tanpa penjelasan.

6. Untuk struktur/kanopi:
   bedakan dengan jelas:
   - rangka utama
   - rangka sekunder
   - tiang
   - bracing/pengaku
   - penutup

7. Jangan menyatakan sebuah struktur "aman" hanya berdasarkan
   perkiraan sederhana. Jika diperlukan verifikasi struktur,
   jelaskan bahwa hasil tersebut adalah estimasi awal dan
   perlu verifikasi engineer/insinyur struktur.

8. Jika data beban belum diberikan:
   jangan mengarang beban angin, beban hidup, atau beban penutup.
   Gunakan asumsi hanya jika pengguna meminta estimasi dan
   nyatakan asumsi tersebut secara eksplisit.

9. Untuk ukuran yang tidak terlihat atau tidak dapat ditentukan:
   jangan mengarang.

10. Untuk pekerjaan bengkel:
    prioritaskan jawaban yang bisa langsung dipakai untuk
    produksi, pemotongan, pengukuran, dan estimasi material.

CODING:
- berikan kode yang dapat dijalankan
- jangan menghilangkan bagian penting dari kode pengguna
- jika memperbaiki kode, jelaskan bagian yang berubah
- gunakan praktik yang aman dan sederhana

OUTPUT TEKNIS:
Jika cocok, gunakan format:

ASUMSI
1. ...
2. ...

PERHITUNGAN
...

CUTTING LIST
1. Batang 1: ...
2. Batang 2: ...

TOTAL
- Kebutuhan: ...
- Waste: ...
- Sisa: ...

CATATAN
...

PRINSIP ROUTER:
- Sistem memilih AI berdasarkan jenis tugas.
- Prioritas utama adalah provider GRATIS.
- Jangan sengaja memakai model berbayar.
- Jika provider gratis gagal, rate limit, timeout,
  unavailable, atau error server, otomatis lanjut
  ke provider gratis berikutnya.

JANGAN PERNAH:
- menampilkan API key
- menampilkan token
- menampilkan password
- menampilkan secret
- membocorkan rahasia sistem
"""


# ============================================================
# AI CLIENTS
# ============================================================

gemini = (
    genai.Client(api_key=GEMINI_KEY)
    if GEMINI_KEY
    else None
)


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
# EPHEMERAL MEMORY
# ============================================================

memory = {}

MAX_MEMORY = 20


def history(uid):
    return memory.setdefault(
        uid,
        []
    )


def remember(uid, role, content):
    history(uid).append(
        {
            "role": role,
            "content": content,
        }
    )

    memory[uid] = history(uid)[
        -MAX_MEMORY:
    ]


def build_messages(uid, text, task):

    task_hint = {

        "coding": """
TUGAS CODING.

Prioritaskan:
- kode yang dapat dijalankan
- ketepatan sintaks
- debugging
- struktur program
- solusi praktis

Jika pengguna memberikan kode,
analisis kode tersebut sebelum mengubahnya.
""",

        "reasoning": """
TUGAS REASONING.

Analisis masalah secara sistematis.
Jangan langsung membuat kesimpulan.
Periksa kemungkinan penyebab dan berikan
kesimpulan paling masuk akal.
""",

        "technical": """
TUGAS TEKNIK/MANUFAKTUR.

Prioritaskan:
- ukuran
- material
- rangka
- fabrikasi
- cutting list
- jumlah batang
- sambungan
- efisiensi material
- asumsi teknik

Untuk perhitungan rangka, bedakan:
rangka utama, rangka sekunder, tiang,
dan pengaku.

Jangan mengarang data yang tidak diberikan.

Jika pengguna meminta jumlah batang 6 meter,
buat kombinasi pemotongan per batang dan
hitung sisa material.
""",

        "math": """
TUGAS MATEMATIKA.

Hitung dengan teliti.
Tampilkan rumus penting.
Tampilkan satuan.
Periksa kembali hasil sebelum menjawab.
""",

        "creative": """
TUGAS KREATIF.

Buat hasil yang siap digunakan,
praktis, menarik, dan sesuai tujuan.
""",

        "general": """
TUGAS UMUM.

Jawab langsung, jelas, dan berguna.
""",

    }.get(task, "")

    return [
        {
            "role": "system",
            "content": (
                SYSTEM
                + "\n\n"
                + task_hint
            ),
        }
    ] + history(uid) + [
        {
            "role": "user",
            "content": text,
        }
    ]


# ============================================================
# TASK CLASSIFIER
# ============================================================

def classify_task(text):

    t = (
        text or ""
    ).lower()

    coding = [
        "python",
        "javascript",
        "typescript",
        "php",
        "html",
        "css",
        "sql",
        "api",
        "coding",
        "kode",
        "program",
        "programming",
        "bug",
        "error",
        "debug",
        "github",
        "vercel",
        "function",
        "import ",
        "async ",
        "def ",
    ]

    technical = [
        "tenda",
        "kanopi",
        "rangka",
        "hollow",
        "pipa",
        "baja",
        "las",
        "fabrikasi",
        "manufaktur",
        "produksi",
        "material",
        "plat",
        "besi",
        "aluminium",
        "konstruksi",
        "ukuran",
        "dimensi",
        "pagar",
        "bengkel",
        "welding",
        "engineering",
        "cutting list",
        "potongan batang",
        "batang 6 meter",
        "rangka utama",
        "rangka sekunder",
    ]

    reasoning = [
        "analisis",
        "analisa",
        "kenapa",
        "mengapa",
        "bandingkan",
        "perbandingan",
        "strategi",
        "logika",
        "alasan",
        "evaluasi",
        "pecahkan",
        "solusi terbaik",
        "reasoning",
    ]

    math = [
        "hitung",
        "perhitungan",
        "berapa",
        "rumus",
        "luas",
        "volume",
        "persentase",
        "matematika",
        "kg",
        "meter",
        "mm",
        "cm",
        "m2",
        "m²",
    ]

    creative = [
        "caption",
        "iklan",
        "promosi",
        "slogan",
        "desain",
        "buatkan gambar",
        "ide konten",
        "copywriting",
    ]

    # CODING paling tinggi
    if any(
        x in t
        for x in coding
    ):
        return "coding"

    # TECHNICAL harus sebelum math.
    # Contoh:
    # "berapa batang hollow untuk kanopi"
    # harus masuk technical,
    # bukan math.
    if any(
        x in t
        for x in technical
    ):
        return "technical"

    if any(
        x in t
        for x in math
    ):
        return "math"

    if any(
        x in t
        for x in reasoning
    ):
        return "reasoning"

    if any(
        x in t
        for x in creative
    ):
        return "creative"

    return "general"


# ============================================================
# OPENROUTER
# ============================================================

def call_openrouter(
    uid,
    text,
    task,
):

    if not openrouter:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum tersedia."
        )

    r = openrouter.chat.completions.create(
        model=OPENROUTER_FREE_MODEL,
        messages=build_messages(
            uid,
            text,
            task,
        ),
        max_tokens=4096,
        extra_headers={
            "HTTP-Referer":
                "https://designmanufaktur.vercel.app",
            "X-Title":
                "Designmanufaktur Super AI Agent",
        },
    )

    answer = (
        r.choices[0].message.content
        or ""
    )

    if not answer.strip():
        raise RuntimeError(
            "OpenRouter Free mengembalikan "
            "jawaban kosong."
        )

    selected_model = (
        getattr(
            r,
            "model",
            None,
        )
        or OPENROUTER_FREE_MODEL
    )

    return (
        answer,
        selected_model,
    )


# ============================================================
# GEMINI
# ============================================================

def call_gemini(
    uid,
    text,
    task,
):

    if not gemini:
        raise RuntimeError(
            "GEMINI_API_KEY belum tersedia."
        )

    task_hint = {

        "coding":
            "Berikan kode yang dapat dijalankan "
            "dan jelaskan perubahan penting.",

        "reasoning":
            "Analisis masalah secara teliti "
            "sebelum memberi kesimpulan.",

        "technical":
            """
Gunakan pertimbangan teknik/manufaktur yang praktis.

Untuk perhitungan material:
- jangan mengarang ukuran
- tampilkan asumsi
- hitung panjang
- buat cutting list
- hitung waste
- bedakan rangka utama dan sekunder
""",

        "math":
            "Hitung secara teliti dan tunjukkan asumsi.",

        "creative":
            "Buat hasil kreatif yang siap digunakan.",

        "general":
            "Jawab langsung dan jelas.",

    }.get(
        task,
        "",
    )

    prompt = (
        SYSTEM
        + "\n\n"
        + task_hint
        + "\n\n"
    )

    for m in history(uid):

        prompt += (
            f"{m['role']}: "
            f"{m['content']}\n"
        )

    prompt += (
        f"user: {text}"
    )

    r = gemini.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
    )

    answer = (
        r.text
        or ""
    )

    if not answer.strip():
        raise RuntimeError(
            "Gemini mengembalikan "
            "jawaban kosong."
        )

    return (
        answer,
        GEMINI_CHAT_MODEL,
    )


# ============================================================
# GROQ
# ============================================================

def call_groq(
    uid,
    text,
    task,
):

    if not groq:
        raise RuntimeError(
            "GROQ_API_KEY belum tersedia."
        )

    if task == "coding":

        model = GROQ_CODING_MODEL

    elif task in (
        "reasoning",
        "math",
    ):

        model = GROQ_REASONING_MODEL

    else:

        model = GROQ_FAST_MODEL

    r = groq.chat.completions.create(
        model=model,
        messages=build_messages(
            uid,
            text,
            task,
        ),
        max_tokens=4096,
    )

    answer = (
        r.choices[0].message.content
        or ""
    )

    if not answer.strip():
        raise RuntimeError(
            "Groq mengembalikan "
            "jawaban kosong."
        )

    return (
        answer,
        model,
    )


# ============================================================
# SMART CHAT ROUTER — FINAL
# ============================================================

def chat_router(
    uid,
    text,
):

    task = classify_task(
        text
    )

    log.info(
        "TASK=%s | text=%s",
        task,
        text[:120],
    )

    errors = []

    # --------------------------------------------------------
    # PRIORITAS PROVIDER
    #
    # TECHNICAL:
    # OpenRouter Free → Gemini → Groq
    #
    # CODING/REASONING/MATH:
    # OpenRouter Free → Groq → Gemini
    #
    # GENERAL/CREATIVE:
    # OpenRouter Free → Gemini → Groq
    # --------------------------------------------------------

    if task == "technical":

        providers = [

            (
                "OpenRouter Free",
                lambda:
                    call_openrouter(
                        uid,
                        text,
                        task,
                    ),
            ),

            (
                "Gemini",
                lambda:
                    call_gemini(
                        uid,
                        text,
                        task,
                    ),
            ),

            (
                "Groq Free Tier",
                lambda:
                    call_groq(
                        uid,
                        text,
                        task,
                    ),
            ),

        ]

    elif task in (
        "coding",
        "reasoning",
        "math",
    ):

        providers = [

            (
                "OpenRouter Free",
                lambda:
                    call_openrouter(
                        uid,
                        text,
                        task,
                    ),
            ),

            (
                "Groq Free Tier",
                lambda:
                    call_groq(
                        uid,
                        text,
                        task,
                    ),
            ),

            (
                "Gemini",
                lambda:
                    call_gemini(
                        uid,
                        text,
                        task,
                    ),
            ),

        ]

    else:

        providers = [

            (
                "OpenRouter Free",
                lambda:
                    call_openrouter(
                        uid,
                        text,
                        task,
                    ),
            ),

            (
                "Gemini",
                lambda:
                    call_gemini(
                        uid,
                        text,
                        task,
                    ),
            ),

            (
                "Groq Free Tier",
                lambda:
                    call_groq(
                        uid,
                        text,
                        task,
                    ),
            ),

        ]

    # --------------------------------------------------------
    # AUTO FALLBACK
    # --------------------------------------------------------

    for provider_name, fn in providers:

        try:

            log.info(
                "TRY PROVIDER | "
                "task=%s | provider=%s",
                task,
                provider_name,
            )

            answer, model = fn()

            if (
                not answer
                or not answer.strip()
            ):
                raise RuntimeError(
                    "Provider mengembalikan "
                    "jawaban kosong."
                )

            log.info(
                "CHAT SUCCESS | "
                "task=%s | provider=%s | model=%s",
                task,
                provider_name,
                model,
            )

            return (
                answer,
                provider_name,
                model,
                task,
            )

        except Exception as e:

            error_text = str(e)

            errors.append(
                f"{provider_name}: "
                f"{error_text[:300]}"
            )

            temporary_error = any(
                code in error_text
                for code in (
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "timeout",
                    "Timeout",
                    "temporarily",
                    "UNAVAILABLE",
                    "Unavailable",
                    "rate limit",
                    "Rate limit",
                    "high demand",
                )
            )

            if temporary_error:

                log.warning(
                    "PROVIDER TEMPORARILY "
                    "UNAVAILABLE | "
                    "provider=%s | error=%s",
                    provider_name,
                    error_text[:300],
                )

            else:

                log.error(
                    "PROVIDER FAILED | "
                    "provider=%s | error=%s",
                    provider_name,
                    error_text[:300],
                )

            # SELALU lanjut provider berikutnya.
            continue

    # --------------------------------------------------------
    # SEMUA PROVIDER GAGAL
    # --------------------------------------------------------

    log.error(
        "ALL FREE PROVIDERS FAILED | "
        "task=%s | errors=%s",
        task,
        " | ".join(errors),
    )

    raise RuntimeError(
        "Semua provider AI GRATIS untuk "
        f"kategori {task} sedang tidak tersedia. "
        "Sistem sudah mencoba seluruh "
        "fallback gratis."
    )


# ============================================================
# TELEGRAM API
# ============================================================

async def tg(
    method,
    data,
):

    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "TELEGRAM_TOKEN belum diatur."
        )

    async with httpx.AsyncClient(
        timeout=180
    ) as client:

        r = await client.post(
            (
                "https://api.telegram.org/"
                f"bot{TELEGRAM_TOKEN}/"
                f"{method}"
            ),
            json=data,
        )

        r.raise_for_status()

        result = r.json()

    if not result.get("ok"):

        raise RuntimeError(
            str(result)
        )

    return result


# ============================================================
# TELEGRAM FILE
# ============================================================

async def tg_file(
    file_id
):

    result = await tg(
        "getFile",
        {
            "file_id": file_id
        },
    )

    path = (
        result["result"]["file_path"]
    )

    async with httpx.AsyncClient(
        timeout=180
    ) as client:

        r = await client.get(
            (
                "https://api.telegram.org/"
                f"file/bot{TELEGRAM_TOKEN}/"
                f"{path}"
            )
        )

        r.raise_for_status()

    return (
        r.content,
        path,
    )


# ============================================================
# TELEGRAM SEND TEXT
# ============================================================

async def send_text(
    chat_id,
    text,
):

    text = (
        text
        or "Tidak ada jawaban."
    )

    # Telegram memiliki batas pesan.
    # Pecah menjadi beberapa pesan.
    for i in range(
        0,
        len(text),
        3900,
    ):

        await tg(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text[
                    i:i + 3900
                ],
            },
        )


# ============================================================
# SEND PHOTO
# ============================================================

async def send_photo(
    chat_id,
    data,
):

    async with httpx.AsyncClient(
        timeout=180
    ) as client:

        r = await client.post(
            (
                "https://api.telegram.org/"
                f"bot{TELEGRAM_TOKEN}/"
                "sendPhoto"
            ),
            data={
                "chat_id":
                    str(chat_id)
            },
            files={
                "photo": (
                    "image.png",
                    data,
                    "image/png",
                )
            },
        )

        r.raise_for_status()


# ============================================================
# SEND VIDEO
# ============================================================

async def send_video(
    chat_id,
    data,
):

    async with httpx.AsyncClient(
        timeout=300
    ) as client:

        r = await client.post(
            (
                "https://api.telegram.org/"
                f"bot{TELEGRAM_TOKEN}/"
                "sendVideo"
            ),
            data={
                "chat_id":
                    str(chat_id)
            },
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

def analyze_image(
    data,
    mime,
    prompt,
):

    if not gemini:

        raise RuntimeError(
            "Gemini belum dikonfigurasi."
        )

    errors = []

    # --------------------------------------------------------
    # GEMINI VISION
    # --------------------------------------------------------

    try:

        r = gemini.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=data,
                    mime_type=mime,
                ),
                (
                    SYSTEM
                    + "\n\n"
                    + prompt
                ),
            ],
        )

        if r.text:

            return (
                r.text,
                GEMINI_CHAT_MODEL,
            )

    except Exception as e:

        errors.append(
            "Gemini Vision: "
            + str(e)[:220]
        )

    # --------------------------------------------------------
    # OPENROUTER VISION FALLBACK
    # --------------------------------------------------------

    if openrouter:

        try:

            b64 = (
                base64.b64encode(
                    data
                ).decode()
            )

            content = [

                {
                    "type": "text",
                    "text": (
                        SYSTEM
                        + "\n\n"
                        + prompt
                    ),
                },

                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                            (
                                f"data:{mime};"
                                f"base64,{b64}"
                            )
                    },
                },

            ]

            r = (
                openrouter
                .chat
                .completions
                .create(
                    model=OPENROUTER_FREE_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    max_tokens=4096,
                )
            )

            answer = (
                r.choices[0]
                .message
                .content
                or ""
            )

            if answer.strip():

                return (
                    answer,
                    getattr(
                        r,
                        "model",
                        OPENROUTER_FREE_MODEL,
                    ),
                )

        except Exception as e:

            errors.append(
                "OpenRouter Free Vision: "
                + str(e)[:220]
            )

    raise RuntimeError(
        "Semua provider vision gratis gagal: "
        + " | ".join(errors)
    )


# ============================================================
# GEMINI VIDEO ANALYSIS
# ============================================================

def analyze_video(
    data,
    mime,
    prompt,
):

    if not gemini:

        raise RuntimeError(
            "Gemini diperlukan "
            "untuk analisis video."
        )

    uploaded = gemini.files.upload(
        file=types.Part.from_bytes(
            data=data,
            mime_type=mime,
        )
    )

    for _ in range(60):

        f = gemini.files.get(
            name=uploaded.name
        )

        state = getattr(
            getattr(
                f,
                "state",
                None,
            ),
            "name",
            "",
        )

        if state == "ACTIVE":

            uploaded = f
            break

        if state == "FAILED":

            raise RuntimeError(
                "Gemini gagal "
                "memproses video."
            )

        time.sleep(2)

    else:

        raise RuntimeError(
            "Video belum siap diproses."
        )

    result = (
        gemini
        .models
        .generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[
                uploaded,
                (
                    SYSTEM
                    + "\n\n"
                    + prompt
                ),
            ],
        )
    )

    return (
        result.text
        or ""
    )


# ============================================================
# IMAGE GENERATION
# FREE ONLY
# ============================================================

def pollinations_image(
    prompt
):

    if not POLLINATIONS_ENABLED:

        raise RuntimeError(
            "Pollinations tidak diaktifkan."
        )

    if not POLLINATIONS_KEY:

        raise RuntimeError(
            "POLLINATIONS_API_KEY "
            "belum tersedia."
        )

    from urllib.parse import quote

    url = (
        f"{POLLINATIONS_BASE_URL}/image/"
        f"{quote(prompt, safe='')}"
        f"?model="
        f"{quote(POLLINATIONS_IMAGE_MODEL)}"
        f"&width=1024"
        f"&height=1024"
    )

    with httpx.Client(
        timeout=300
    ) as client:

        r = client.get(
            url,
            headers={
                "Authorization":
                    (
                        "Bearer "
                        f"{POLLINATIONS_KEY}"
                    ),
                "Accept":
                    "image/png,image/jpeg,*/*",
            },
        )

        if r.status_code >= 400:

            raise RuntimeError(
                f"Pollinations HTTP "
                f"{r.status_code}: "
                f"{r.text[:400]}"
            )

        if not r.content:

            raise RuntimeError(
                "Pollinations "
                "mengembalikan data kosong."
            )

        return r.content


def generate_image(
    prompt
):

    if POLLINATIONS_ENABLED:

        return (
            pollinations_image(
                prompt
            ),
            "Pollinations",
        )

    raise RuntimeError(
        "Generate gambar GRATIS "
        "belum tersedia. "
        "Aktifkan "
        "POLLINATIONS_ENABLED=true "
        "dan "
        "POLLINATIONS_API_KEY."
    )


# ============================================================
# COMMAND ARGUMENT
# ============================================================

def command_arg(
    text
):

    parts = text.split(
        maxsplit=1
    )

    return (
        parts[1].strip()
        if len(parts) > 1
        else ""
    )


# ============================================================
# HANDLE TELEGRAM UPDATE
# ============================================================

async def handle(
    update
):

    message = update.get(
        "message"
    )

    if not message:
        return

    chat_id = (
        message
        .get("chat", {})
        .get("id")
    )

    uid = str(
        message
        .get("from", {})
        .get(
            "id",
            chat_id,
        )
    )

    text = (
        message.get(
            "text",
            ""
        )
        or ""
    )

    caption = (
        message.get(
            "caption",
            ""
        )
        or ""
    )

    # ========================================================
    # /START
    # ========================================================

    if text.startswith(
        "/start"
    ):

        await send_text(
            chat_id,
            """🤖 Designmanufaktur Super AI Agent aktif.

🧠 Smart Multi-AI Router
💰 FREE-FIRST
🖼️ Gemini Vision
🎥 Gemini Video Analysis
🎨 Free Image Generation

Chat biasa → AI dipilih sesuai jenis tugas.

Teknik/Manufaktur:
OpenRouter Free → Gemini → Groq

Coding:
OpenRouter Free → Groq → Gemini

Reasoning/Math:
OpenRouter Free → Groq → Gemini

General/Creative:
OpenRouter Free → Gemini → Groq

Jika provider gagal → otomatis fallback.

/model → status AI
/reset → hapus memory sesi
/gambar <prompt> → generate gambar gratis
/video → analisis video""",
        )

        return

    # ========================================================
    # /RESET
    # ========================================================

    if text.startswith(
        "/reset"
    ):

        memory.pop(
            uid,
            None,
        )

        await send_text(
            chat_id,
            "✅ Memory sesi dihapus.",
        )

        return

    # ========================================================
    # /MODEL
    # ========================================================

    if text.startswith(
        "/model"
    ):

        await send_text(
            chat_id,
            f"""STATUS SUPER AI AGENT

Gemini:
{'✅ AKTIF' if gemini else '❌ TIDAK AKTIF'}

OpenRouter FREE:
{'✅ AKTIF' if openrouter else '❌ TIDAK AKTIF'}

Groq FREE-TIER:
{'✅ AKTIF' if groq else '❌ TIDAK AKTIF'}

Gemini Chat:
{GEMINI_CHAT_MODEL}

OpenRouter:
{OPENROUTER_FREE_MODEL}

Groq Coding:
{GROQ_CODING_MODEL}

Groq Reasoning:
{GROQ_REASONING_MODEL}

Groq Fast:
{GROQ_FAST_MODEL}

Gemini Vision:
{GEMINI_CHAT_MODEL}

Image Generation FREE:
{'Pollinations ✅' if POLLINATIONS_ENABLED and POLLINATIONS_KEY else 'Tidak aktif'}

ROUTING:

Technical/Manufacturing
→ OpenRouter Free
→ Gemini
→ Groq

Coding/Reasoning/Math
→ OpenRouter Free
→ Groq
→ Gemini

General/Creative
→ OpenRouter Free
→ Gemini
→ Groq

Vision
→ Gemini
→ OpenRouter Free

PAID MODEL ROUTING:
DISABLED""",
        )

        return

    # ========================================================
    # /GAMBAR
    # ========================================================

    if text.startswith(
        "/gambar"
    ):

        prompt = command_arg(
            text
        )

        if not prompt:

            await send_text(
                chat_id,
                "Contoh:\n"
                "/gambar pagar "
                "minimalis hitam modern",
            )

            return

        await send_text(
            chat_id,
            "🎨 Memilih generator "
            "gambar GRATIS...",
        )

        try:

            data, provider = (
                await asyncio.to_thread(
                    generate_image,
                    prompt,
                )
            )

            await send_photo(
                chat_id,
                data,
            )

            await send_text(
                chat_id,
                f"✅ Gambar dibuat "
                f"oleh {provider}.",
            )

        except Exception as e:

            log.exception(
                "image generation failed"
            )

            await send_text(
                chat_id,
                "❌ Generate gambar "
                "gratis gagal.\n"
                + str(e)[:700],
            )

        return

    # ========================================================
    # VIDEO
    # ========================================================

    if message.get(
        "video"
    ):

        await send_text(
            chat_id,
            "🎥 Sedang menganalisis "
            "video...",
        )

        try:

            data, path = (
                await tg_file(
                    message[
                        "video"
                    ][
                        "file_id"
                    ]
                )
            )

            if len(data) > (
                20 * 1024 * 1024
            ):

                await send_text(
                    chat_id,
                    "❌ Video lebih "
                    "dari 20 MB.",
                )

                return

            mime = (
                "video/quicktime"
                if path.lower().endswith(
                    ".mov"
                )
                else "video/mp4"
            )

            answer = (
                await asyncio.to_thread(
                    analyze_video,
                    data,
                    mime,
                    caption
                    or
                    (
                        "Analisa video ini "
                        "secara detail. "
                        "Jelaskan objek, "
                        "proses, kondisi, "
                        "masalah yang terlihat, "
                        "dan saran praktis."
                    ),
                )
            )

            await send_text(
                chat_id,
                answer,
            )

        except Exception as e:

            log.exception(
                "video analysis failed"
            )

            await send_text(
                chat_id,
                "❌ Analisis video "
                "gagal.\n"
                + str(e)[:700],
            )

        return

    # ========================================================
    # PHOTO
    # ========================================================

    if message.get(
        "photo"
    ):

        await send_text(
            chat_id,
            "🖼️ Gemini Vision "
            "sedang menganalisis "
            "gambar...",
        )

        try:

            data, path = (
                await tg_file(
                    message[
                        "photo"
                    ][-1][
                        "file_id"
                    ]
                )
            )

            mime = (
                mimetypes.guess_type(
                    path
                )[0]
                or "image/jpeg"
            )

            prompt = (
                caption
                or
                """
Analisa gambar ini secara detail.

Jika terkait manufaktur, bengkel las,
tenda, pagar, fabrikasi, konstruksi,
atau produk custom:

- jelaskan objek
- jelaskan komponen
- jelaskan fungsi
- jelaskan kondisi
- perkirakan hanya data yang benar-benar
  dapat diperkirakan dari visual
- jelaskan masalah yang terlihat
- berikan saran praktis

Jangan mengarang ukuran atau data
yang tidak terlihat pada gambar.
"""
            )

            answer, model = (
                await asyncio.to_thread(
                    analyze_image,
                    data,
                    mime,
                    prompt,
                )
            )

            await send_text(
                chat_id,
                answer,
            )

            log.info(
                "VISION SUCCESS | "
                "model=%s",
                model,
            )

        except Exception as e:

            log.exception(
                "image analysis failed"
            )

            await send_text(
                chat_id,
                "❌ Analisis gambar "
                "gagal.\n"
                + str(e)[:700],
            )

        return

    # ========================================================
    # NORMAL CHAT
    # ========================================================

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

        (
            answer,
            provider,
            model,
            task,
        ) = await asyncio.to_thread(
            chat_router,
            uid,
            text,
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
            "CHAT DONE | "
            "task=%s | "
            "provider=%s | "
            "model=%s",
            task,
            provider,
            model,
        )

    except Exception as e:

        log.exception(
            "chat failed"
        )

        await send_text(
            chat_id,
            "❌ Semua AI GRATIS "
            "gagal untuk request ini.\n\n"
            + str(e)[:700],
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "ok": True,
        "service":
            "Designmanufaktur "
            "Super AI Agent",

        "free_only": True,

        "providers": {
            "gemini":
                bool(gemini),

            "openrouter_free":
                bool(openrouter),

            "groq_free_tier":
                bool(groq),
        },

        "models": {
            "gemini":
                GEMINI_CHAT_MODEL,

            "openrouter":
                OPENROUTER_FREE_MODEL,

            "groq_coding":
                GROQ_CODING_MODEL,

            "groq_reasoning":
                GROQ_REASONING_MODEL,

            "groq_fast":
                GROQ_FAST_MODEL,
        },
    }


# ============================================================
# /API
# ============================================================

@app.get("/api")
async def api_root():

    return await root()


# ============================================================
# WEBHOOK IMPLEMENTATION
# ============================================================

async def webhook_impl(
    request: Request,
    x_telegram_bot_api_secret_token:
        Optional[str],
):

    if (
        WEBHOOK_SECRET
        and
        x_telegram_bot_api_secret_token
        != WEBHOOK_SECRET
    ):

        raise HTTPException(
            status_code=403,
            detail="Invalid webhook secret",
        )

    update = (
        await request.json()
    )

    await handle(
        update
    )

    return {
        "ok": True
    }


# ============================================================
# TELEGRAM WEBHOOK
# ============================================================

@app.post(
    "/api/webhook"
)
async def webhook(
    request: Request,
    x_telegram_bot_api_secret_token:
        Optional[str] = Header(
            default=None
        ),
):

    return await webhook_impl(
        request,
        x_telegram_bot_api_secret_token,
    )


# ============================================================
# LEGACY ROOT WEBHOOK
# ============================================================

@app.post("/")
async def root_post(
    request: Request,
    x_telegram_bot_api_secret_token:
        Optional[str] = Header(
            default=None
        ),
):

    return await webhook_impl(
        request,
        x_telegram_bot_api_secret_token,
    )