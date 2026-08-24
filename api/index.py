import asyncio
import base64
import mimetypes
import os
import time
import logging
import re
import math
import json
from urllib.parse import quote
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
    title="Designmanufaktur Super AI Agent + Civil Calculator"
)


# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
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

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_FREE_MODEL = "openrouter/free"

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
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

POLLINATIONS_BASE_URL = "https://gen.pollinations.ai"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM = """
Kamu adalah Designmanufaktur Super AI Agent.

Kamu adalah asisten AI praktis untuk:

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
- pekerjaan sipil
- beton
- pondasi
- sloof
- kolom
- balok
- plat lantai
- dinding
- plesteran
- acian
- besi tulangan
- cutting list
- optimasi material
- perhitungan panjang
- estimasi jumlah batang
- optimasi cutting stock

PRINSIP UTAMA:

1. Jangan mengarang data yang belum diberikan.
2. Jika data tidak tersedia, nyatakan dengan jelas bahwa data belum diberikan.
3. Bedakan data input, asumsi, hasil perhitungan, dan rekomendasi.
4. Untuk pekerjaan struktur, jangan menyatakan "aman" tanpa data dan pemeriksaan engineering yang memadai.
5. Untuk perhitungan material, lakukan sanity check.
6. Periksa kembali jumlah, panjang, volume, dan satuan.
7. Jangan double-counting material.
8. Untuk cutting list, perhatikan panjang stock, jumlah potongan, sisa material, dan waste.
9. Jika ada beberapa kemungkinan interpretasi, jelaskan interpretasinya secara singkat.
10. Jawaban harus praktis dan mudah dipahami.

Untuk cutting stock:

- Hitung kebutuhan bersih.
- Hitung minimum teoritis.
- Susun pola pemotongan.
- Pastikan setiap pola <= panjang stock.
- Hitung total stock.
- Hitung sisa setiap batang.
- Bedakan reusable offcut dan true waste.
- Lakukan sanity check terhadap seluruh jumlah potongan.

FORMAT PERHITUNGAN:

Input:
- jumlah
- panjang
- lebar
- tinggi
- tebal
- panjang stock

Asumsi:
- hanya jika memang diperlukan.

Perhitungan:
- tampilkan rumus penting.
- gunakan satuan konsisten.

Hasil:
- jumlah material.
- panjang total.
- jumlah stock.
- sisa material.

Sanity check:
- cocokkan kembali seluruh potongan dengan kebutuhan awal.

Untuk pertanyaan umum:
jawab singkat, jelas, dan langsung.

Untuk pertanyaan teknik:
gunakan struktur:

1. Data input
2. Asumsi
3. Analisis
4. Hasil
5. Sanity check
6. Catatan

Untuk desain struktur:
hasil perhitungan material bukan pengganti desain engineer.

Kamu juga mendukung perintah:

/model
/reset
/gambar <prompt>

Dan perintah perhitungan lokal seperti:

/potong

Untuk /potong, jangan mengandalkan AI jika perhitungan lokal tersedia.
Gunakan calculator engine terlebih dahulu.
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
# MEMORY — PERSISTENT GITHUB V2
# ============================================================

memory = {}

MAX_MEMORY = 20

# Memory BARU.
# Memory lama users/... TIDAK digunakan.
MEMORY_DIR = "memory_v2"

# Menggunakan Environment Variable yang SUDAH ADA
# di Vercel.
GITHUB_TOKEN = os.getenv(
    "GITHUB_TOKEN",
    ""
)

GITHUB_REPO = os.getenv(
    "GITHUB_REPO",
    "Dinal7297/designmanufaktur-memory"
)

GITHUB_BRANCH = os.getenv(
    "GITHUB_BRANCH",
    "main"
)

GITHUB_API = "https://api.github.com"


def history(uid):
    uid = str(uid)
    return memory.setdefault(uid, [])


def _memory_path(uid):
    return f"{MEMORY_DIR}/{str(uid)}.json"


async def load_persistent_memory(uid):
    """
    Membaca memory baru dari GitHub.

    Memory lama di users/... tidak disentuh
    dan tidak digunakan.
    """

    uid = str(uid)

    if not GITHUB_TOKEN or not GITHUB_REPO:
        memory.setdefault(uid, [])
        return

    url = (
        f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/"
        f"{quote(_memory_path(uid), safe='/')}"
    )

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:

        async with httpx.AsyncClient(
            timeout=15.0
        ) as client:

            response = await client.get(
                url,
                headers=headers,
                params={
                    "ref": GITHUB_BRANCH
                },
            )

        if response.status_code == 404:

            memory[uid] = []

            return

        response.raise_for_status()

        data = response.json()

        encoded = data.get(
            "content",
            ""
        )

        if not encoded:

            memory[uid] = []

            return

        raw = base64.b64decode(
            encoded.replace("\n", "")
        ).decode("utf-8")

        saved = json.loads(raw)

        if isinstance(saved, dict):

            saved = saved.get(
                "memory",
                []
            )

        if not isinstance(
            saved,
            list
        ):

            saved = []

        memory[uid] = saved[
            -MAX_MEMORY:
        ]

        log.info(
            "MEMORY LOAD OK | uid=%s | items=%s",
            uid,
            len(memory[uid]),
        )

    except Exception as e:

        log.warning(
            "MEMORY LOAD FAILED | uid=%s | %s",
            uid,
            e,
        )

        memory.setdefault(
            uid,
            []
        )


async def save_persistent_memory(uid):
    """
    Menyimpan memory baru ke GitHub.

    Menggunakan GitHub Contents API.
    """

    uid = str(uid)

    if not GITHUB_TOKEN or not GITHUB_REPO:

        log.warning(
            "MEMORY SAVE SKIPPED | "
            "GITHUB_TOKEN/GITHUB_REPO belum tersedia"
        )

        return

    payload_memory = history(uid)[
        -MAX_MEMORY:
    ]

    raw = json.dumps(
        {
            "user_id": uid,
            "memory": payload_memory,
            "updated_at": int(
                time.time()
            ),
        },
        ensure_ascii=False,
        indent=2,
    )

    encoded = base64.b64encode(
        raw.encode("utf-8")
    ).decode("ascii")

    url = (
        f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/"
        f"{quote(_memory_path(uid), safe='/')}"
    )

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    for attempt in range(3):

        try:

            async with httpx.AsyncClient(
                timeout=15.0
            ) as client:

                current = await client.get(
                    url,
                    headers=headers,
                    params={
                        "ref": GITHUB_BRANCH
                    },
                )

                body = {
                    "message":
                        f"memory: update {uid}",
                    "content": encoded,
                    "branch": GITHUB_BRANCH,
                }

                if current.status_code == 200:

                    body["sha"] = (
                        current.json()
                        .get("sha")
                    )

                elif current.status_code != 404:

                    current.raise_for_status()

                saved = await client.put(
                    url,
                    headers=headers,
                    json=body,
                )

            if saved.status_code in (
                200,
                201,
            ):

                log.info(
                    "MEMORY SAVE OK | uid=%s",
                    uid,
                )

                return

            if (
                saved.status_code == 409
                and attempt < 2
            ):

                await asyncio.sleep(
                    0.5
                )

                continue

            saved.raise_for_status()

        except Exception as e:

            if attempt < 2:

                await asyncio.sleep(
                    0.5
                )

                continue

            log.warning(
                "MEMORY SAVE FAILED | uid=%s | %s",
                uid,
                e,
            )


def remember(
    uid,
    role,
    content,
):

    uid = str(uid)

    history(uid).append(
        {
            "role": role,
            "content": content,
        }
    )

    memory[uid] = history(uid)[
        -MAX_MEMORY:
    ]


def build_messages(
    uid,
    text,
    task,
):

    task_hint = {

        "coding": """
TUGAS CODING.

Prioritaskan:
- kode yang dapat dijalankan
- ketepatan sintaks
- debugging
- struktur program
- solusi praktis
""",

        "reasoning": """
TUGAS REASONING.

Analisis masalah secara sistematis.
Periksa kemungkinan penyebab.
Validasi kesimpulan sebelum menjawab.
""",

        "technical": """
TUGAS TEKNIK/MANUFAKTUR.

Prioritaskan:
- data input
- asumsi yang benar-benar diperlukan
- perhitungan
- daftar material
- cutting list
- efisiensi material
- sanity check

Jangan mengarang data yang belum diberikan.

Untuk struktur:
hasil material bukan pengganti desain engineer.
""",

        "civil": """
TUGAS SIPIL.

Gunakan perhitungan yang teliti.

Jangan mengarang data.
Jika data belum diberikan,
nyatakan data belum ditentukan.

Bedakan:
estimasi material
dengan
desain struktur.

Untuk struktur:
jangan menyatakan aman
tanpa perhitungan engineering.
""",

        "math": """
TUGAS MATEMATIKA.

Hitung dengan teliti.
Periksa satuan.
Lakukan sanity check.
""",

        "creative": """
TUGAS KREATIF.

Buat hasil yang praktis,
jelas, dan siap digunakan.
""",

        "general": """
TUGAS UMUM.

Jawab langsung,
jelas,
dan tidak bertele-tele.
""",
    }.get(
        task,
        ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM
                + "\n\n"
                + task_hint
            ),
        }
    ]

    for m in history(uid):

        messages.append(
            {
                "role": m["role"],
                "content": m["content"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": text,
        }
    )

    return messages


# ============================================================
# TELEGRAM API
# ============================================================

async def tg(
    method,
    payload=None,
):

    if not TELEGRAM_TOKEN:

        raise RuntimeError(
            "TELEGRAM_TOKEN belum tersedia."
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/{method}"
    )

    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        response = await client.post(
            url,
            json=payload or {},
        )

    response.raise_for_status()

    data = response.json()

    if not data.get("ok"):

        raise RuntimeError(
            str(data)
        )

    return data


async def send_text(
    chat_id,
    text,
):

    if not text:

        return

    max_len = 4000

    if len(text) <= max_len:

        await tg(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
            },
        )

        return

    for i in range(
        0,
        len(text),
        max_len,
    ):

        await tg(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text[
                    i:i + max_len
                ],
            },
        )


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    if not text:

        return ""

    return str(text).strip()


def detect_task(text):

    t = normalize_text(
        text
    ).lower()

    coding = [
        "python",
        "javascript",
        "typescript",
        "html",
        "css",
        "json",
        "api",
        "coding",
        "kode",
        "program",
        "script",
        "bug",
        "error code",
        "debug",
        "github",
        "vercel",
    ]

    civil = [
        "beton",
        "cor",
        "coran",
        "pondasi",
        "footplat",
        "sloof",
        "kolom",
        "balok",
        "plat beton",
        "plat lantai",
        "dinding",
        "tembok",
        "bata",
        "batako",
        "plester",
        "acian",
        "galian",
        "urugan",
        "besi tulangan",
        "tulangan",
        "begel",
        "bendrat",
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
        "purlin",
        "pengaku",
        "tiang",
        "balok utama",
        "sambungan",
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

    math_keywords = [
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

    if any(
        x in t
        for x in coding
    ):

        return "coding"

    if any(
        x in t
        for x in civil
    ):

        return "civil"

    if any(
        x in t
        for x in technical
    ):

        return "technical"

    if any(
        x in t
        for x in reasoning
    ):

        return "reasoning"

    if any(
        x in t
        for x in math_keywords
    ):

        return "math"

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
            "OpenRouter Free mengembalikan jawaban kosong."
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
            "Berikan kode yang dapat dijalankan.",

        "reasoning":
            "Analisis masalah secara teliti sebelum kesimpulan.",

        "technical":
            "Gunakan pertimbangan teknik dan manufaktur yang praktis.",

        "civil":
            """
Gunakan perhitungan sipil secara teliti.

Jangan mengarang data.
Jika data belum diberikan, nyatakan data belum ditentukan.

Bedakan:
estimasi material
dengan
desain struktur.

Untuk struktur:
jangan menyatakan aman tanpa perhitungan engineering.
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

    answer = r.text or ""

    if not answer.strip():

        raise RuntimeError(
            "Gemini mengembalikan jawaban kosong."
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
            "Groq mengembalikan jawaban kosong."
        )

    return (
        answer,
        model,
    )


# ============================================================
# ROUTER
# ============================================================

def chat_router(
    uid,
    text,
):

    task = detect_task(
        text
    )

    errors = []

    if gemini:

        try:

            answer, model = call_gemini(
                uid,
                text,
                task,
            )

            return (
                answer,
                "Gemini",
                model,
                task,
            )

        except Exception as e:

            errors.append(
                f"Gemini: {e}"
            )

    if groq:

        try:

            answer, model = call_groq(
                uid,
                text,
                task,
            )

            return (
                answer,
                "Groq",
                model,
                task,
            )

        except Exception as e:

            errors.append(
                f"Groq: {e}"
            )

    if openrouter:

        try:

            answer, model = call_openrouter(
                uid,
                text,
                task,
            )

            return (
                answer,
                "OpenRouter FREE",
                model,
                task,
            )

        except Exception as e:

            errors.append(
                f"OpenRouter: {e}"
            )

    raise RuntimeError(
        " | ".join(errors)
        or "Tidak ada AI yang aktif."
    )


# ============================================================
# LOCAL CUTTING CALCULATOR
# ============================================================

def parse_cutting_command(text):

    raw = text.strip()

    if not raw.lower().startswith(
        "/potong"
    ):

        return None

    body = raw[
        len("/potong"):
    ].strip()

    if not body:

        return None

    stock_match = re.search(
        r"stock\s*=\s*([0-9]+(?:[.,][0-9]+)?)",
        body,
        re.I,
    )

    if not stock_match:

        return None

    stock = float(
        stock_match.group(1)
        .replace(",", ".")
    )

    body = re.sub(
        r"stock\s*=\s*[0-9]+(?:[.,][0-9]+)?",
        "",
        body,
        flags=re.I,
    )

    pieces = []

    for item in re.findall(
        r"(\d+)\s*x\s*([0-9]+(?:[.,][0-9]+)?)",
        body,
        re.I,
    ):

        qty = int(
            item[0]
        )

        length = float(
            item[1]
            .replace(",", ".")
        )

        for _ in range(qty):

            pieces.append(
                length
            )

    if not pieces:

        return None

    return (
        stock,
        pieces,
    )


def optimize_cutting(
    stock,
    pieces,
):

    pieces = sorted(
        pieces,
        reverse=True,
    )

    bins = []

    for length in pieces:

        placed = False

        best_index = None
        best_remaining = None

        for i, remaining in enumerate(
            bins
        ):

            if length <= remaining + 1e-9:

                new_remaining = (
                    remaining
                    - length
                )

                if (
                    best_remaining is None
                    or new_remaining
                    < best_remaining
                ):

                    best_index = i
                    best_remaining = (
                        new_remaining
                    )

        if best_index is not None:

            bins[
                best_index
            ]["cuts"].append(
                length
            )

            bins[
                best_index
            ]["remaining"] = (
                best_remaining
            )

            placed = True

        if not placed:

            bins.append(
                {
                    "cuts": [length],
                    "remaining":
                        stock - length,
                }
            )

    return bins


def format_cutting_result(
    stock,
    pieces,
    bins,
):

    total_required = sum(
        pieces
    )

    total_stock = (
        len(bins) * stock
    )

    total_remaining = (
        total_stock
        - total_required
    )

    efficiency = (
        total_required
        / total_stock
        * 100
        if total_stock
        else 0
    )

    lines = []

    lines.append(
        "✂️ CUTTING LIST"
    )

    lines.append("")

    lines.append(
        f"Stock: {stock:g} m"
    )

    lines.append(
        f"Total kebutuhan: {total_required:g} m"
    )

    lines.append(
        f"Jumlah stock: {len(bins)} batang"
    )

    lines.append(
        f"Total tersedia: {total_stock:g} m"
    )

    lines.append(
        f"Total sisa: {total_remaining:g} m"
    )

    lines.append(
        f"Efisiensi: {efficiency:.2f}%"
    )

    lines.append("")

    for i, b in enumerate(
        bins,
        start=1,
    ):

        cuts = " + ".join(
            f"{x:g} m"
            for x in b["cuts"]
        )

        lines.append(
            f"Batang {i}: "
            f"{cuts}"
        )

        lines.append(
            f"  Sisa: "
            f"{b['remaining']:g} m"
        )

    lines.append("")

    lines.append(
        "✅ SANITY CHECK"
    )

    requested_counts = {}

    for x in pieces:

        key = round(
            x,
            6
        )

        requested_counts[key] = (
            requested_counts.get(
                key,
                0
            ) + 1
        )

    result_counts = {}

    for b in bins:

        for x in b["cuts"]:

            key = round(
                x,
                6
            )

            result_counts[key] = (
                result_counts.get(
                    key,
                    0
                ) + 1
            )

    if requested_counts == result_counts:

        lines.append(
            "✅ Semua jumlah potongan sesuai input."
        )

    else:

        lines.append(
            "❌ Jumlah potongan tidak cocok."
        )

    if all(
        sum(b["cuts"]) <= stock + 1e-9
        for b in bins
    ):

        lines.append(
            "✅ Tidak ada batang yang melebihi stock."
        )

    else:

        lines.append(
            "❌ Ada pola pemotongan melebihi stock."
        )

    return "\n".join(
        lines
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def get_mime_type(
    filename,
):

    mime, _ = mimetypes.guess_type(
        filename or ""
    )

    return mime or "image/jpeg"


async def download_telegram_file(
    file_id,
):

    info = await tg(
        "getFile",
        {
            "file_id": file_id,
        },
    )

    file_path = (
        info
        .get("result", {})
        .get("file_path")
    )

    if not file_path:

        raise RuntimeError(
            "Telegram file_path tidak ditemukan."
        )

    url = (
        f"https://api.telegram.org/"
        f"file/bot{TELEGRAM_TOKEN}/"
        f"{file_path}"
    )

    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        response = await client.get(
            url
        )

    response.raise_for_status()

    mime = (
        get_mime_type(
            file_path
        )
    )

    return (
        response.content,
        mime,
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    data,
    mime,
    prompt,
):

    if not gemini:

        raise RuntimeError(
            "GEMINI_API_KEY belum tersedia."
        )

    image_part = types.Part.from_bytes(
        data=data,
        mime_type=mime,
    )

    response = (
        gemini.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[
                image_part,
                prompt,
            ],
        )
    )

    answer = (
        response.text
        or ""
    )

    if not answer.strip():

        raise RuntimeError(
            "Gemini vision mengembalikan jawaban kosong."
        )

    return (
        answer,
        GEMINI_CHAT_MODEL,
    )


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image_gemini(
    prompt,
):

    if not gemini:

        raise RuntimeError(
            "GEMINI_API_KEY belum tersedia."
        )

    response = (
        gemini.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=[
                    "TEXT",
                    "IMAGE",
                ]
            ),
        )
    )

    image_data = None
    text_parts = []

    for part in (
        response.candidates[0]
        .content.parts
    ):

        if getattr(
            part,
            "text",
            None,
        ):

            text_parts.append(
                part.text
            )

        inline = getattr(
            part,
            "inline_data",
            None,
        )

        if inline:

            image_data = (
                inline.data
            )

    if not image_data:

        raise RuntimeError(
            "Gemini tidak menghasilkan gambar."
        )

    return (
        image_data,
        "image/png",
        "\n".join(text_parts),
    )


async def send_photo_bytes(
    chat_id,
    data,
    mime="image/png",
    caption="",
):

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendPhoto"
    )

    files = {
        "photo": (
            "image.png",
            data,
            mime,
        )
    }

    form = {
        "chat_id": str(chat_id),
    }

    if caption:

        form["caption"] = caption[:1024]

    async with httpx.AsyncClient(
        timeout=120.0
    ) as client:

        response = await client.post(
            url,
            data=form,
            files=files,
        )

    response.raise_for_status()


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "ok",
        "service":
            "Designmanufaktur Super AI Agent",
        "memory":
            "github_v2",
        "calculator":
            "active",
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "telegram":
            bool(TELEGRAM_TOKEN),
        "gemini":
            bool(gemini),
        "openrouter":
            bool(openrouter),
        "groq":
            bool(groq),
        "github_memory":
            bool(
                GITHUB_TOKEN
                and GITHUB_REPO
            ),
    }


# ============================================================
# WEBHOOK
# ============================================================

@app.post("/api")
async def telegram_webhook(
    request: Request,
):

    body = await request.json()

    return await process_update(
        body
    )


@app.post("/api/index")
async def telegram_webhook_index(
    request: Request,
):

    body = await request.json()

    return await process_update(
        body
    )


# ============================================================
# PROCESS UPDATE
# ============================================================

async def process_update(
    update,
):

    try:

        message = (
            update.get(
                "message"
            )
            or update.get(
                "edited_message"
            )
            or {}
        )

        chat = message.get(
            "chat",
            {}
        )

        chat_id = chat.get(
            "id"
        )

        if chat_id is None:

            return {
                "ok": True
            }

        from_user = (
            message.get(
                "from",
                {}
            )
        )

        uid = str(
            from_user.get(
                "id",
                chat_id
            )
        )

        text = normalize_text(
            message.get(
                "text",
                ""
            )
        )

        # ====================================================
        # PHOTO
        # ====================================================

        photos = message.get(
            "photo"
        )

        if photos:

            try:

                largest = photos[-1]

                data, mime = (
                    await download_telegram_file(
                        largest.get(
                            "file_id"
                        )
                    )
                )

                caption = normalize_text(
                    message.get(
                        "caption",
                        ""
                    )
                )

                prompt = caption or """
Analisis gambar ini secara teknis.

Jelaskan:
- objek
- komponen
- ukuran yang terlihat
- kondisi
- kemungkinan fungsi
- masalah yang terlihat
- saran praktis

Jangan mengarang ukuran yang tidak terlihat.
"""

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
                    "VISION SUCCESS | model=%s",
                    model,
                )

            except Exception as e:

                log.exception(
                    "image analysis failed"
                )

                await send_text(
                    chat_id,
                    "❌ Analisis gambar gagal.\n"
                    + str(e)[:700],
                )

            return {
                "ok": True
            }

        # ====================================================
        # NORMAL CHAT
        # ====================================================

        if not text:

            return {
                "ok": True
            }

        # ====================================================
        # /START
        # ====================================================

        if text.startswith(
            "/start"
        ):

            await send_text(
                chat_id,
                """
🤖 Designmanufaktur Super AI Agent

Siap membantu:

• Kalkulator sipil
• Cutting list
• Optimasi material
• Kanopi
• Tenda
• Rangka
• Hollow
• Pipa
• Fabrikasi
• Manufaktur
• Teknik

Perintah:

/model
/reset
/potong 5x2, 4x3, 2.5 stock=6
/gambar <prompt>

Memory baru sudah menggunakan penyimpanan persisten.
""",
            )

            return {
                "ok": True
            }

        # ====================================================
        # /RESET
        # ====================================================

        if text.startswith(
            "/reset"
        ):

            memory.pop(
                uid,
                None,
            )

            await save_persistent_memory(
                uid
            )

            await send_text(
                chat_id,
                "✅ Memory baru dihapus.",
            )

            return {
                "ok": True
            }

        # ====================================================
        # /MODEL
        # ====================================================

        if text.startswith(
            "/model"
        ):

            await send_text(
                chat_id,
                f"""
🤖 STATUS SUPER AI AGENT

Gemini:
{'✅ AKTIF' if gemini else '❌ TIDAK AKTIF'}

OpenRouter FREE:
{'✅ AKTIF' if openrouter else '❌ TIDAK AKTIF'}

Groq FREE:
{'✅ AKTIF' if groq else '❌ TIDAK AKTIF'}

🏗️ CIVIL CALCULATOR

✅ ACTIVE
Local Calculation Engine

Kemampuan:

• Beton
• Sloof
• Kolom
• Balok
• Plat
• Footplat
• Pondasi
• Dinding
• Plester
• Acian
• Galian
• Urugan
• Besi

🧠 MODEL

Gemini:
{GEMINI_CHAT_MODEL}

OpenRouter:
{OPENROUTER_FREE_MODEL}

Groq Coding:
{GROQ_CODING_MODEL}

Groq Reasoning:
{GROQ_REASONING_MODEL}

Groq Fast:
{GROQ_FAST_MODEL}

💾 MEMORY

GitHub:
{'✅ AKTIF' if GITHUB_TOKEN and GITHUB_REPO else '❌ TIDAK AKTIF'}

Folder:
{MEMORY_DIR}

💰 PAID MODEL ROUTING
DISABLED
""",
            )

            return {
                "ok": True
            }

        # ====================================================
        # /POTONG
        # ====================================================

        cutting = parse_cutting_command(
            text
        )

        if cutting:

            stock, pieces = cutting

            if stock <= 0:

                await send_text(
                    chat_id,
                    "❌ Panjang stock harus lebih dari 0.",
                )

                return {
                    "ok": True
                }

            if any(
                p > stock + 1e-9
                for p in pieces
            ):

                await send_text(
                    chat_id,
                    "❌ Ada potongan yang lebih panjang daripada stock.",
                )

                return {
                    "ok": True
                }

            bins = optimize_cutting(
                stock,
                pieces,
            )

            result = format_cutting_result(
                stock,
                pieces,
                bins,
            )

            await send_text(
                chat_id,
                result,
            )

            return {
                "ok": True
            }

        # ====================================================
        # /GAMBAR
        # ====================================================

        if text.startswith(
            "/gambar"
        ):

            prompt = text[
                len("/gambar"):
            ].strip()

            if not prompt:

                await send_text(
                    chat_id,
                    "Gunakan:\n/gambar <deskripsi gambar>",
                )

                return {
                    "ok": True
                }

            try:

                await tg(
                    "sendChatAction",
                    {
                        "chat_id": chat_id,
                        "action": "upload_photo",
                    },
                )

                data, mime, caption = (
                    await asyncio.to_thread(
                        generate_image_gemini,
                        prompt,
                    )
                )

                await send_photo_bytes(
                    chat_id,
                    data,
                    mime,
                    caption,
                )

            except Exception as e:

                log.exception(
                    "image generation failed"
                )

                await send_text(
                    chat_id,
                    "❌ Pembuatan gambar gagal.\n"
                    + str(e)[:700],
                )

            return {
                "ok": True
            }

        # ====================================================
        # LOAD MEMORY BARU
        # ====================================================

        await load_persistent_memory(
            uid
        )

        # ====================================================
        # TYPING
        # ====================================================

        await tg(
            "sendChatAction",
            {
                "chat_id": chat_id,
                "action": "typing",
            },
        )

        # ====================================================
        # AI ROUTER
        # ====================================================

        try:

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

        except Exception as e:

            log.exception(
                "chat router failed"
            )

            await send_text(
                chat_id,
                "❌ Semua AI GRATIS gagal untuk request ini.\n\n"
                + str(e)[:700],
            )

            return {
                "ok": True
            }

        # ====================================================
        # SAVE CONVERSATION
        # ====================================================

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

        # ====================================================
        # SAVE MEMORY PERSISTENT
        # ====================================================

        await save_persistent_memory(
            uid
        )

        # ====================================================
        # SEND ANSWER
        # ====================================================

        await send_text(
            chat_id,
            answer,
        )

        log.info(
            "CHAT DONE | "
            "uid=%s | task=%s | provider=%s | model=%s",
            uid,
            task,
            provider,
            model,
        )

        return {
            "ok": True
        }

    except Exception as e:

        log.exception(
            "process_update failed"
        )

        try:

            if (
                "chat_id" in locals()
                and chat_id
            ):

                await send_text(
                    chat_id,
                    "❌ Terjadi error pada bot.\n"
                    + str(e)[:700],
                )

        except Exception:

            pass

        return {
            "ok": False,
            "error": str(e)[:500],
        }


# ============================================================
# VERCEL HANDLER COMPATIBILITY
# ============================================================

handler = app