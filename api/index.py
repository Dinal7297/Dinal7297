index.py — Designmanufaktur Super AI Agent + Civil Calculator

import asyncio
import base64
import mimetypes
import os
import time
import logging
import re
import math
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
- pekerjaan sipil
- desain produk custom
- cutting list
- estimasi material
- engineering
- perhitungan teknis
- coding/programming
- bisnis
- konten dan pemasaran

============================================================
BAHASA
============================================================

Jawab dalam Bahasa Indonesia kecuali pengguna meminta bahasa lain.

============================================================
GAYA JAWABAN
============================================================

- langsung ke inti
- praktis
- jelas
- tidak bertele-tele
- nyaman dibaca di HP
- gunakan tabel hanya jika benar-benar membantu
- gunakan satuan yang jelas
- hasil harus bisa dipakai untuk pekerjaan lapangan
- jangan membuat jawaban terlihat rumit tanpa alasan

============================================================
FORMAT TELEGRAM
============================================================

Jawaban akan dikirim melalui Telegram.

WAJIB membuat jawaban nyaman dibaca pada layar HP.

HINDARI:

- bold
- italic
- heading Markdown
- tabel dengan karakter |
- dekorasi simbol berlebihan
- tanda bintang berulang
- garis pemisah panjang

Gunakan emoji seperlunya.

Contoh heading:

📋 DATA
⚙️ ASUMSI
🧮 PERHITUNGAN
📐 KEBUTUHAN SIPIL
✂️ CUTTING LIST
🔍 VALIDASI
📊 RINGKASAN
📝 CATATAN
🎯 KESIMPULAN

Gunakan daftar:

• Item pertama
• Item kedua
• Item ketiga

============================================================
ATURAN AKURASI
============================================================

1. Jangan mengarang ukuran, harga, material, beban, kapasitas,
   koefisien, atau spesifikasi yang tidak diberikan pengguna.

2. Jika data belum tersedia, tulis:

"Data belum ditentukan."

3. Untuk perhitungan:

- tuliskan data
- tuliskan asumsi
- tuliskan rumus penting
- hitung hasil
- lakukan validasi ulang
- tuliskan hasil akhir
- gunakan satuan konsisten

4. Jangan menganggap hasil benar hanya karena operasi matematikanya
   terlihat benar.

5. Sebelum memberikan jawaban akhir, periksa ulang seluruh angka.

============================================================
PERHITUNGAN SIPIL
============================================================

Kamu juga berfungsi sebagai asisten estimasi pekerjaan sipil.

Jenis pekerjaan yang dapat dianalisis:

- luas bangunan
- luas dinding
- volume beton
- volume pondasi
- volume sloof
- volume kolom
- volume balok
- volume lantai
- volume pekerjaan tanah
- kebutuhan semen
- kebutuhan pasir
- kebutuhan split/kerikil
- kebutuhan beton
- kebutuhan bata merah
- kebutuhan batako
- kebutuhan hebel
- kebutuhan mortar
- kebutuhan plester
- kebutuhan acian
- kebutuhan keramik
- kebutuhan nat
- kebutuhan besi tulangan
- kebutuhan begel/sengkang
- estimasi berat besi
- estimasi jumlah batang besi
- kebutuhan kawat bendrat
- estimasi material pekerjaan sipil

============================================================
ATURAN PERHITUNGAN SIPIL
============================================================

Selalu bedakan:

1. HASIL MATEMATIS

Contoh:

Panjang × Lebar × Tebal = Volume.

2. ASUMSI MATERIAL

Contoh:

Volume beton = 5 m³.

Kebutuhan semen/pasir/split tidak boleh dibuat seolah-olah
pasti apabila metode campuran atau koefisien belum ditentukan.

Jika koefisien tidak diberikan:

- nyatakan bahwa koefisien belum ditentukan
- gunakan asumsi hanya jika pengguna meminta estimasi
- tuliskan asumsi tersebut secara jelas

Jangan menyamarkan asumsi sebagai data pasti.

============================================================
BETON
============================================================

Untuk beton:

Volume = panjang × lebar × tebal.

Semua dimensi harus dikonversi ke meter.

Contoh:

Panjang = 10 m
Lebar = 5 m
Tebal = 0,10 m

Volume:

10 × 5 × 0,10 = 5 m³

Jika pengguna meminta kebutuhan semen, pasir, dan split:

Jangan mengarang komposisi.

Tanyakan atau nyatakan metode:

- mutu beton
- mix design
- rasio campuran
- koefisien pekerjaan
- atau asumsi estimasi

Jika menggunakan asumsi, tampilkan secara eksplisit.

============================================================
PONDASI
============================================================

Untuk pondasi menerus:

Volume = panjang × lebar × tinggi.

Jika bentuk pondasi berbeda:

- persegi
- trapesium
- batu kali
- foot plate
- cakar ayam
- pile cap

identifikasi bentuk terlebih dahulu.

Untuk trapesium:

Luas penampang =
((sisi bawah + sisi atas) / 2) × tinggi

Volume =
luas penampang × panjang.

============================================================
SLOOF / BALOK
============================================================

Volume:

panjang × lebar × tinggi.

Contoh:

Sloof 15 cm × 20 cm × 30 m

Konversi:

0,15 × 0,20 × 30

= 0,90 m³

============================================================
KOLOM
============================================================

Volume satu kolom:

lebar × panjang × tinggi.

Total:

volume satu kolom × jumlah kolom.

============================================================
TULANGAN
============================================================

Jika pengguna memberikan:

- diameter
- jumlah batang
- panjang
- jumlah komponen

hitung panjang total terlebih dahulu.

Contoh:

4 batang D10
panjang sloof 30 m

Panjang tulangan utama:

4 × 30 = 120 m

Jika pengguna meminta berat:

Gunakan rumus pendekatan berat besi:

berat per meter ≈ d² / 162

d = diameter dalam mm.

Contoh D10:

10² / 162
= 0,617 kg/m

Total berat:

panjang total × berat per meter.

============================================================
BEGEL / SENGKANG
============================================================

Jika diketahui:

panjang komponen
jarak begel

Jumlah begel secara pendekatan:

ceil(panjang / jarak) + 1

Namun posisi begel pertama dan terakhir harus tetap
disesuaikan dengan detail gambar kerja.

Jangan mengklaim sebagai detail struktur final.

============================================================
BATA / BATAKO / HEBEL
============================================================

Jika pengguna memberikan luas dinding dan ukuran material:

Luas dinding =
panjang × tinggi

Luas satu unit =
panjang unit × tinggi unit

Jumlah unit teoritis:

luas dinding / luas satu unit

Tambahkan faktor waste hanya jika diberikan atau diminta.

Jika menggunakan waste:

jumlah akhir =
jumlah teoritis × (1 + persentase waste)

============================================================
PLESER
============================================================

Luas plester:

panjang dinding × tinggi dinding × jumlah sisi.

Jika bukaan pintu/jendela diberikan:

luas bersih =
luas dinding - luas bukaan.

Volume plester:

luas × ketebalan plester.

============================================================
ACIAN
============================================================

Luas acian mengikuti luas permukaan yang akan diaci.

Jika pengguna meminta kebutuhan semen:

gunakan koefisien yang diberikan pengguna
atau nyatakan asumsi.

============================================================
KERAMIK
============================================================

Luas lantai:

panjang × lebar.

Jumlah keramik:

luas lantai / luas satu keramik.

Tambahkan waste sesuai kebutuhan pemasangan jika diberikan.

Contoh:

lantai = 20 m²
keramik = 60 × 60 cm

luas satu keramik:

0,60 × 0,60
= 0,36 m²

jumlah teoritis:

20 / 0,36
= 55,56

dibulatkan menjadi 56 keping sebelum waste.

============================================================
SATUAN
============================================================

Konversi:

1 m = 100 cm
1 m = 1000 mm
1 m² = 1.000.000 mm²
1 m³ = 1.000 liter

Untuk perhitungan volume:

mm → m
cm → m

sebelum perhitungan.

============================================================
VALIDASI SIPIL
============================================================

Sebelum menjawab:

CHECK 1
Apakah semua dimensi menggunakan satuan konsisten?

CHECK 2
Apakah luas benar?

CHECK 3
Apakah volume benar?

CHECK 4
Apakah jumlah komponen benar?

CHECK 5
Apakah pembulatan dilakukan dengan benar?

CHECK 6
Apakah waste dipisahkan dari kebutuhan bersih?

CHECK 7
Apakah asumsi material disebutkan?

CHECK 8
Apakah ada data yang sebenarnya belum diberikan?

CHECK 9
Apakah perhitungan tidak double counting?

CHECK 10
Jika menyangkut struktur, apakah sudah diberikan peringatan
bahwa hasil bukan pengganti desain engineer?

============================================================
KESELAMATAN STRUKTUR
============================================================

Jangan menyatakan:

"aman"

"hancur"

"pasti kuat"

atau klaim struktural final

hanya berdasarkan perhitungan sederhana.

Jika menyangkut:

- pondasi
- kolom
- balok
- sloof
- dak
- struktur baja
- kanopi
- bangunan
- beban gempa
- beban angin
- kapasitas tanah

jelaskan bahwa hasil adalah estimasi awal apabila data
struktur lengkap belum tersedia.

Untuk desain final diperlukan verifikasi engineer/insinyur
struktur dan data lapangan yang sesuai.

============================================================
CUTTING LIST
============================================================

Jika pengguna meminta cutting list, kebutuhan batang,
optimasi material, atau potongan dari batang standar:

WAJIB:

1. Identifikasi panjang batang standar.
2. Identifikasi semua potongan.
3. Hitung total kebutuhan.
4. Hitung batas bawah teoritis.
5. Lakukan bin packing.
6. Pastikan setiap batang <= kapasitas.
7. Validasi jumlah setiap potongan.
8. Validasi material dibeli.
9. Validasi material terpakai.
10. Validasi total sisa.
11. Bedakan TRUE WASTE dan REUSABLE OFFCUT.
12. Periksa double counting.
13. Tandai komponen yang membutuhkan sambungan.

Jangan menggunakan:

ceil(total / panjang batang)

sebagai jawaban final.

Itu hanya batas bawah teoritis.

============================================================
TRUE WASTE DAN REUSABLE OFFCUT
============================================================

TRUE WASTE:

Sisa yang tidak dapat digunakan untuk kebutuhan potongan
yang sedang dihitung.

REUSABLE OFFCUT:

Sisa material yang masih berguna dan dapat disimpan
untuk pekerjaan lain.

Jangan menyamakan keduanya.

============================================================
PRIVASI
============================================================

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
# MEMORY
# ============================================================

memory = {}
MAX_MEMORY = 20


def history(uid):
    return memory.setdefault(uid, [])


def remember(uid, role, content):

    history(uid).append({
        "role": role,
        "content": content,
    })

    memory[uid] = history(uid)[-MAX_MEMORY:]


# ============================================================
# TASK HINT
# ============================================================

TASK_HINTS = {

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
Jangan langsung membuat kesimpulan.
Validasi kesimpulan sebelum menjawab.
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

Untuk cutting list:
WAJIB validasi jumlah potongan,
kapasitas batang, material, sisa,
true waste, reusable offcut,
dan double counting.
""",

    "civil": """
TUGAS PERHITUNGAN SIPIL.

Prioritaskan:
- luas
- volume
- beton
- pondasi
- sloof
- kolom
- balok
- tulangan
- begel
- bata
- batako
- hebel
- plester
- acian
- keramik
- semen
- pasir
- split
- kebutuhan material

WAJIB:
1. Identifikasi semua data.
2. Konversi satuan.
3. Tuliskan asumsi.
4. Gunakan rumus yang sesuai.
5. Hitung hasil.
6. Validasi ulang.
7. Bedakan kebutuhan bersih dan waste.
8. Jangan mengarang koefisien material.
9. Jika koefisien tidak tersedia, nyatakan asumsi atau minta data.
10. Untuk struktur, jangan menyatakan aman tanpa verifikasi engineer.

HASIL HARUS NYAMAN DIBACA DI TELEGRAM.
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
}


def build_messages(uid, text, task):

    task_hint = TASK_HINTS.get(
        task,
        TASK_HINTS["general"]
    )

    return [
        {
            "role": "system",
            "content": SYSTEM + "\n\n" + task_hint,
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

    t = (text or "").lower()

    coding = [
        "python", "javascript", "typescript", "php",
        "html", "css", "sql", "api", "coding",
        "kode", "program", "programming", "bug",
        "error", "debug", "github", "vercel",
        "function", "import ", "async ", "def ",
    ]

    civil = [
        "sipil", "beton", "cor ", "cor ",
        "pondasi", "pondasi batu", "foot plate",
        "cakar ayam", "sloof", "kolom", "balok",
        "dak", "dak beton", "lantai beton",
        "tulangan", "besi tulangan", "begel",
        "sengkang", "bendrat",
        "bata", "bata merah", "batako", "hebel",
        "dinding", "pasangan bata", "mortar",
        "plester", "plesteran", "acian",
        "keramik", "ubin", "lantai",
        "semen", "pasir", "split", "kerikil",
        "volume beton", "volume pondasi",
        "kebutuhan material sipil",
        "material bangunan", "bangunan",
        "rumah", "ruko", "gedung",
        "galian", "urugan", "timbunan",
        "bekisting",
    ]

    technical = [
        "tenda", "kanopi", "rangka", "hollow", "pipa",
        "baja", "las", "fabrikasi", "manufaktur",
        "produksi", "material", "plat", "besi",
        "aluminium", "konstruksi", "ukuran", "dimensi",
        "pagar", "bengkel", "welding", "engineering",
        "cutting list", "potongan batang",
        "batang 6 meter", "rangka utama",
        "rangka sekunder", "purlin", "pengaku",
        "tiang", "balok utama", "sambungan",
    ]

    reasoning = [
        "analisis", "analisa", "kenapa", "mengapa",
        "bandingkan", "perbandingan", "strategi",
        "logika", "alasan", "evaluasi", "pecahkan",
        "solusi terbaik", "reasoning",
    ]

    math = [
        "hitung", "perhitungan", "berapa", "rumus",
        "luas", "volume", "persentase", "matematika",
        "kg", "meter", "mm", "cm", "m2", "m²",
    ]

    creative = [
        "caption", "iklan", "promosi", "slogan",
        "desain", "buatkan gambar", "ide konten",
        "copywriting",
    ]

    if any(x in t for x in coding):
        return "coding"

    if any(x in t for x in civil):
        return "civil"

    if any(x in t for x in technical):
        return "technical"

    if any(x in t for x in math):
        return "math"

    if any(x in t for x in reasoning):
        return "reasoning"

    if any(x in t for x in creative):
        return "creative"

    return "general"


# ============================================================
# OPENROUTER
# ============================================================

def call_openrouter(uid, text, task):

    if not openrouter:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum tersedia."
        )

    r = openrouter.chat.completions.create(
        model=OPENROUTER_FREE_MODEL,
        messages=build_messages(uid, text, task),
        max_tokens=4096,
        extra_headers={
            "HTTP-Referer":
                "https://designmanufaktur.vercel.app",
            "X-Title":
                "Designmanufaktur Super AI Agent",
        },
    )

    answer = r.choices[0].message.content or ""

    if not answer.strip():
        raise RuntimeError(
            "OpenRouter Free mengembalikan jawaban kosong."
        )

    selected_model = (
        getattr(r, "model", None)
        or OPENROUTER_FREE_MODEL
    )

    return answer, selected_model


# ============================================================
# GEMINI
# ============================================================

def call_gemini(uid, text, task):

    if not gemini:
        raise RuntimeError(
            "GEMINI_API_KEY belum tersedia."
        )

    task_hint = TASK_HINTS.get(
        task,
        TASK_HINTS["general"]
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

    prompt += f"user: {text}"

    r = gemini.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=prompt,
    )

    answer = r.text or ""

    if not answer.strip():
        raise RuntimeError(
            "Gemini mengembalikan jawaban kosong."
        )

    return answer, GEMINI_CHAT_MODEL


# ============================================================
# GROQ
# ============================================================

def call_groq(uid, text, task):

    if not groq:
        raise RuntimeError(
            "GROQ_API_KEY belum tersedia."
        )

    if task == "coding":
        model = GROQ_CODING_MODEL

    elif task in ("reasoning", "math", "civil"):
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
        raise RuntimeError(
            "Groq mengembalikan jawaban kosong."
        )

    return answer, model


# ============================================================
# SMART CHAT ROUTER
# ============================================================

def chat_router(uid, text):

    task = classify_task(text)

    log.info(
        "TASK=%s | text=%s",
        task,
        text[:120],
    )

    if task in ("technical", "civil"):

        providers = [
            (
                "OpenRouter Free",
                lambda: call_openrouter(
                    uid,
                    text,
                    task,
                ),
            ),
            (
                "Gemini",
                lambda: call_gemini(
                    uid,
                    text,
                    task,
                ),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(
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
                lambda: call_openrouter(
                    uid,
                    text,
                    task,
                ),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(
                    uid,
                    text,
                    task,
                ),
            ),
            (
                "Gemini",
                lambda: call_gemini(
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
                lambda: call_openrouter(
                    uid,
                    text,
                    task,
                ),
            ),
            (
                "Gemini",
                lambda: call_gemini(
                    uid,
                    text,
                    task,
                ),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(
                    uid,
                    text,
                    task,
                ),
            ),
        ]

    errors = []

    for provider_name, fn in providers:

        try:

            log.info(
                "TRY PROVIDER | task=%s | provider=%s",
                task,
                provider_name,
            )

            answer, model = fn()

            if not answer or not answer.strip():
                raise RuntimeError(
                    "Provider mengembalikan jawaban kosong."
                )

            log.info(
                "CHAT SUCCESS | task=%s | provider=%s | model=%s",
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

            log.warning(
                "PROVIDER FAILED | provider=%s | error=%s",
                provider_name,
                error_text[:300],
            )

            continue

    raise RuntimeError(
        "Semua provider AI GRATIS untuk "
        f"kategori {task} sedang tidak tersedia. "
        "Sistem sudah mencoba seluruh fallback gratis."
    )


# ============================================================
# TELEGRAM API
# ============================================================

async def tg(method, data):

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
        raise RuntimeError(str(result))

    return result


# ============================================================
# TELEGRAM FILE
# ============================================================

async def tg_file(file_id):

    result = await tg(
        "getFile",
        {"file_id": file_id},
    )

    path = result["result"]["file_path"]

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

    return r.content, path


# ============================================================
# TELEGRAM RESPONSE FORMATTER
# ============================================================

def clean_telegram_text(text):

    if not text:
        return "Tidak ada jawaban."

    text = str(text).replace(
        "\r\n",
        "\n",
    )

    text = re.sub(
        r"```[a-zA-Z0-9_+\-]*\n?",
        "",
        text,
    )

    text = text.replace(
        "```",
        "",
    )

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = text.replace(
        "**",
        "",
    )

    text = re.sub(
        r"(?<!\w)\*(?!\s)",
        "",
        text,
    )

    text = text.replace(
        "__",
        "",
    )

    text = text.replace(
        "`",
        "",
    )

    text = re.sub(
        r"^\s*[-_*]{3,}\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        if "|" in stripped:

            if re.fullmatch(
                r"[\s|:_\-]+",
                stripped,
            ):
                continue

            cells = [
                cell.strip()
                for cell in stripped.strip("|").split("|")
            ]

            cells = [
                cell
                for cell in cells
                if cell
            ]

            if cells:
                line = " — ".join(cells)

        line = re.sub(
            r"^\s*[-*+]\s+",
            "• ",
            line,
        )

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    heading_emojis = {
        "DATA": "📋",
        "ASUMSI": "⚙️",
        "PERHITUNGAN": "🧮",
        "KEBUTUHAN SIPIL": "📐",
        "CUTTING LIST": "✂️",
        "VALIDASI": "🔍",
        "RINGKASAN": "📊",
        "CATATAN": "📝",
        "HASIL": "✅",
        "KESIMPULAN": "🎯",
        "MATERIAL": "🔩",
        "SAMBUNGAN": "🔧",
        "WASTE": "♻️",
        "TRUE WASTE": "🗑️",
        "REUSABLE OFFCUT": "♻️",
    }

    result = []

    for line in text.split("\n"):

        clean = line.strip()

        if not clean:

            result.append("")
            continue

        upper = clean.upper()

        for heading, emoji in heading_emojis.items():

            if upper == heading:

                if not clean.startswith(
                    emoji
                ):

                    clean = (
                        f"{emoji} {heading}"
                    )

                break

        result.append(clean)

    text = "\n".join(result)

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# SMART TELEGRAM CHUNK
# ============================================================

def split_telegram_message(
    text,
    max_length=3900,
):

    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    paragraphs = text.split(
        "\n\n"
    )

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            current
            + (
                "\n\n"
                if current
                else ""
            )
            + paragraph
        )

        if len(candidate) <= max_length:

            current = candidate
            continue

        if current:

            chunks.append(
                current.strip()
            )

            current = ""

        lines = paragraph.split("\n")

        for line in lines:

            line = line.strip()

            if not line:
                continue

            candidate = (
                current
                + (
                    "\n"
                    if current
                    else ""
                )
                + line
            )

            if len(candidate) <= max_length:

                current = candidate

            else:

                if current:

                    chunks.append(
                        current.strip()
                    )

                current = ""

                while len(line) > max_length:

                    chunks.append(
                        line[:max_length]
                    )

                    line = line[max_length:]

                current = line

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# SEND TEXT
# ============================================================

async def send_text(
    chat_id,
    text,
):

    formatted = clean_telegram_text(
        text
    )

    chunks = split_telegram_message(
        formatted,
        max_length=3900,
    )

    for chunk in chunks:

        await tg(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": chunk,
            },
        )

        if len(chunks) > 1:

            await asyncio.sleep(
                0.25
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
                "chat_id": str(chat_id)
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
                "chat_id": str(chat_id)
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

    try:

        r = gemini.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=data,
                    mime_type=mime,
                ),
                SYSTEM
                + "\n\n"
                + prompt,
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

    if openrouter:

        try:

            b64 = base64.b64encode(
                data
            ).decode()

            content = [
                {
                    "type": "text",
                    "text":
                        SYSTEM
                        + "\n\n"
                        + prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                            f"data:{mime};base64,{b64}"
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
# GEMINI VIDEO
# ============================================================

def analyze_video(
    data,
    mime,
    prompt,
):

    if not gemini:

        raise RuntimeError(
            "Gemini diperlukan untuk analisis video."
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
                "Gemini gagal memproses video."
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
                SYSTEM
                + "\n\n"
                + prompt,
            ],
        )
    )

    return result.text or ""


# ============================================================
# IMAGE GENERATION
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
            "POLLINATIONS_API_KEY belum tersedia."
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
                    f"Bearer {POLLINATIONS_KEY}",
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
                "Pollinations mengembalikan data kosong."
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
        "Generate gambar GRATIS belum tersedia. "
        "Aktifkan POLLINATIONS_ENABLED=true "
        "dan POLLINATIONS_API_KEY."
    )


# ============================================================
# COMMAND ARGUMENT
# ============================================================

def command_arg(text):

    parts = text.split(
        maxsplit=1
    )

    return (
        parts[1].strip()
        if len(parts) > 1
        else ""
    )


# ============================================================
# SIPIL HELP
# ============================================================

def civil_help():

    return """
📐 KALKULATOR KEBUTUHAN SIPIL

Bot dapat membantu menghitung:

• Luas bangunan
• Luas dinding
• Volume beton
• Pondasi
• Sloof
• Kolom
• Balok
• Tulangan
• Begel
• Bata
• Batako
• Hebel
• Plester
• Acian
• Keramik
• Semen
• Pasir
• Split/kerikil
• Estimasi material

Contoh 1:

Hitung volume cor lantai
panjang 10 m
lebar 5 m
tebal 10 cm

Contoh 2:

Hitung volume sloof
panjang 30 m
ukuran 15 x 20 cm

Contoh 3:

Sloof panjang 30 m,
tulangan utama 4D10,
begel D8 jarak 15 cm.
Hitung kebutuhan besinya.

Contoh 4:

Dinding panjang 10 m
tinggi 3 m.
Hitung kebutuhan bata.

Contoh 5:

Lantai 5 x 10 m,
keramik 60 x 60 cm.
Hitung jumlah keramik.

⚠️ Untuk kebutuhan semen,
pasir, split, mortar, dan material
berdasarkan campuran tertentu,
bot akan meminta atau menjelaskan
asumsi/koefisien yang digunakan.

⚠️ Untuk struktur bangunan,
hasil AI adalah estimasi awal,
bukan pengganti perhitungan
engineer/insinyur struktur.
"""


# ============================================================
# HANDLE TELEGRAM UPDATE
# ============================================================

async def handle(update):

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
        .get("id", chat_id)
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
    # START
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

📐 CIVIL CALCULATOR
✅ Beton
✅ Pondasi
✅ Sloof
✅ Kolom
✅ Balok
✅ Tulangan
✅ Begel
✅ Bata
✅ Batako
✅ Hebel
✅ Plester
✅ Acian
✅ Keramik
✅ Estimasi material

✂️ CUTTING LIST
✅ Validasi jumlah potongan
✅ Validasi kapasitas batang
✅ Validasi total material
✅ Validasi sambungan
✅ True Waste
✅ Reusable Offcut
✅ Anti double-counting

Technical/Manufacturing:
OpenRouter Free → Gemini → Groq

Civil:
OpenRouter Free → Gemini → Groq

Coding:
OpenRouter Free → Groq → Gemini

Reasoning/Math:
OpenRouter Free → Groq → Gemini

General/Creative:
OpenRouter Free → Gemini → Groq

Jika provider gagal → otomatis fallback.

Perintah:

/model
/reset
/sipil
/gambar <prompt>
/video

Ketik /sipil untuk contoh perhitungan sipil.""",
        )

        return

    # ========================================================
    # RESET
    # ========================================================

    if text.startswith(
        "/reset"
    ):

        memory.pop(
            uid,
            None
        )

        await send_text(
            chat_id,
            "✅ Memory sesi dihapus.",
        )

        return

    # ========================================================
    # SIPIL
    # ========================================================

    if text.startswith(
        "/sipil"
    ):

        await send_text(
            chat_id,
            civil_help(),
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    if text.startswith(
        "/model"
    ):

        await send_text(
            chat_id,
            f"""🤖 STATUS SUPER AI AGENT

Gemini:
{'✅ AKTIF' if gemini else '❌ TIDAK AKTIF'}

OpenRouter FREE:
{'✅ AKTIF' if openrouter else '❌ TIDAK AKTIF'}

Groq FREE-TIER:
{'✅ AKTIF' if groq else '❌ TIDAK AKTIF'}

🧠 MODEL

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

🔀 ROUTING

Technical/Manufacturing
→ OpenRouter Free
→ Gemini
→ Groq

Civil
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

📐 CIVIL
✅ Volume
✅ Material
✅ Tulangan
✅ Begel
✅ Bata
✅ Plester
✅ Acian
✅ Keramik
✅ Validasi asumsi

✂️ CUTTING LIST
✅ Quantity validation
✅ Capacity validation
✅ Material validation
✅ Connection validation
✅ True waste validation
✅ Reusable offcut validation
✅ Double-count validation

💰 PAID MODEL ROUTING
DISABLED""",
        )

        return

    # ========================================================
    # GAMBAR
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
                """🎨 GENERATE GAMBAR

Contoh:

/gambar pagar minimalis hitam modern""",
            )

            return

        await send_text(
            chat_id,
            "🎨 Memilih generator gambar GRATIS...",
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
                f"✅ Gambar dibuat oleh {provider}.",
            )

        except Exception as e:

            log.exception(
                "image generation failed"
            )

            await send_text(
                chat_id,
                "❌ Generate gambar gratis gagal.\n"
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
            "🎥 Sedang menganalisis video...",
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
                    "❌ Video lebih dari 20 MB.",
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
                    caption or (
                        "Analisa video ini secara detail. "
                        "Jelaskan objek, proses, kondisi, "
                        "masalah yang terlihat, dan saran praktis."
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
                "❌ Analisis video gagal.\n"
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
            "🖼️ Gemini Vision sedang menganalisis gambar...",
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

            prompt = caption or """
Analisa gambar ini secara detail.

Jika terkait manufaktur, bengkel las,
tenda, pagar, fabrikasi, konstruksi,
atau pekerjaan sipil:

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
            "CHAT DONE | task=%s | provider=%s | model=%s",
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
            "❌ Semua AI GRATIS gagal untuk request ini.\n\n"
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
            "Designmanufaktur Super AI Agent",
        "free_only": True,
        "telegram_format":
            "clean_and_mobile_friendly",
        "capabilities": {
            "civil_calculation": True,
            "cutting_list": True,
            "vision": bool(gemini),
            "video": bool(gemini),
            "image_generation":
                POLLINATIONS_ENABLED,
        },
        "providers": {
            "gemini": bool(gemini),
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
# API
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

    update = await request.json()

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