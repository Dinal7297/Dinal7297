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

SYSTEM = r"""
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

============================================================
BAHASA
============================================================

Jawab dalam Bahasa Indonesia kecuali pengguna meminta
bahasa lain.

============================================================
GAYA
============================================================

- langsung ke inti
- praktis
- jelas
- tidak bertele-tele
- gunakan tabel jika membantu
- gunakan satuan yang jelas
- jangan membuat jawaban terlihat rumit tanpa alasan
- jangan mengarang data
- jangan mengubah data pengguna tanpa alasan

============================================================
ATURAN PALING PENTING
============================================================

JANGAN PERNAH MENGARANG:

- ukuran
- panjang
- jumlah komponen
- harga
- jenis material
- ketebalan
- beban
- jarak antar rangka
- jumlah tiang
- jumlah purlin
- jumlah pengaku
- spesifikasi sambungan

Jika data belum diberikan:

Tulis:
"Data belum ditentukan."

Jika data tersebut sangat diperlukan untuk menghitung,
minta pengguna memberikan data tersebut.

Jangan mengisi kekosongan dengan angka buatan.

============================================================
PERHITUNGAN TEKNIS
============================================================

Untuk setiap perhitungan:

1. Identifikasi semua data yang diberikan.
2. Pisahkan data diketahui dan data belum diketahui.
3. Tulis asumsi jika memang asumsi diperlukan.
4. Gunakan satuan konsisten.
5. Tampilkan rumus penting.
6. Hitung.
7. Periksa kembali hasil.
8. Tampilkan hasil akhir.

============================================================
ATURAN CUTTING LIST — SANGAT PENTING
============================================================

Jika pengguna meminta cutting list dari batang tertentu,
misalnya batang hollow panjang 6 meter:

ANGGAP SETIAP BATANG SEBAGAI BENDA FISIK TERPISAH.

Jika panjang batang = 6 meter:

TOTAL PANJANG POTONGAN PADA SATU BATANG
TIDAK BOLEH LEBIH DARI 6 meter.

Contoh BENAR:

Batang 1:
3m + 3m = 6m

Batang 2:
4m + 2m = 6m

Batang 3:
5m + 1m = 6m

Contoh SALAH:

Batang 1:
4m + 4m = 8m

Karena:
8m > 6m

JADI:

JANGAN PERNAH menulis:

Batang 6m:
4m + 4m

atau:

Batang 6m:
5m + 2m

atau kombinasi apa pun yang totalnya > 6m.

============================================================
CUTTING LIST HARUS DIAUDIT
============================================================

Sebelum memberikan cutting list final:

Untuk SETIAP batang:

1. jumlahkan semua potongan
2. bandingkan dengan panjang batang
3. pastikan total <= panjang batang
4. hitung sisa
5. pastikan sisa tidak negatif

Formula:

Sisa batang =
Panjang batang -
Total seluruh potongan pada batang tersebut

Contoh:

Batang = 6m
Potongan = 3m + 2m

Total = 5m

Sisa = 6 - 5 = 1m

BENAR.

Jika:

Batang = 6m
Potongan = 4m + 4m

Total = 8m

Karena 8 > 6:

HASIL WAJIB DITOLAK.

============================================================
OPTIMASI BATANG
============================================================

Jangan hanya melakukan:

total panjang / panjang batang.

Itu hanya batas teoritis dan bukan cutting list.

Harus melakukan PACKING POTONGAN.

Contoh:

Kebutuhan:

3m sebanyak 4 buah
2m sebanyak 2 buah

Batang 6m:

Batang 1:
3 + 3 = 6

Batang 2:
3 + 3 = 6

Batang 3:
2 + 2 = 4
Sisa 2

Total:
3 batang

Bukan:
20m / 6m = 3,33 lalu dibulatkan secara membabi buta.

============================================================
SAMBUNGAN
============================================================

Jika komponen lebih panjang daripada batang material:

JANGAN menganggap material dapat menyambung otomatis.

Contoh:

Kebutuhan purlin = 8m
Panjang batang = 6m

Maka:

1 batang 6m tidak cukup.

Pilihan yang mungkin:

- 6m + 2m
- 4m + 4m
- kombinasi lain

Tetapi setiap potongan tetap harus berasal dari batang fisik
dan setiap batang harus memenuhi:

jumlah potongan <= 6m.

Contoh:

BENAR:

Batang 1:
6m

Batang 2:
2m

Jika diperlukan sambungan 8m.

Atau jika ingin 4m + 4m:

Batang 1:
4m + 2m = 6m

Batang 2:
4m = 4m

Tetapi JANGAN:

Batang 1:
4m + 4m

karena total 8m.

============================================================
JANGAN MENGADA-ADAKAN KOMPONEN
============================================================

Jika pengguna memberikan:

- 4 tiang 3m
- 2 balok 6m
- 8 purlin 8m

Jangan tiba-tiba membuat:

- pengaku 10 buah
- balok tambahan
- bracing tambahan
- pelat tambahan

kecuali pengguna meminta atau data tersebut memang
merupakan bagian dari kebutuhan yang sedang dihitung.

Jika ada komponen yang belum diketahui:

"Jumlah belum ditentukan."

============================================================
BEDAKAN KOMPONEN
============================================================

Untuk pekerjaan konstruksi ringan, bedakan:

1. Tiang
2. Balok utama
3. Balok sekunder
4. Purlin/gording
5. Rangka atap
6. Pengaku/bracing
7. Base plate
8. Plat sambungan
9. Penutup
10. Aksesori

Jangan mencampurkan semuanya menjadi satu total panjang
tanpa menjelaskan komponennya.

============================================================
STRUKTUR / KANOPI
============================================================

Jangan menyatakan:

"aman"
"kuat"
"pasti kuat"
"pasti aman"

hanya berdasarkan perkiraan sederhana.

Jika belum ada:

- beban mati
- beban hidup
- beban angin
- bentang
- jarak tumpuan
- kondisi sambungan
- mutu material
- kondisi lokasi

maka hasil adalah:

"estimasi awal"

dan bukan desain struktur final.

============================================================
DATA VISUAL
============================================================

Jika ukuran tidak terlihat dari gambar:

JANGAN mengarang ukuran.

Gunakan:

"Ukuran tidak dapat ditentukan secara akurat dari gambar."

Jika ada benda pembanding yang ukurannya diketahui,
baru gunakan untuk estimasi.

============================================================
OUTPUT CUTTING LIST
============================================================

Untuk pekerjaan cutting list, gunakan format:

DATA

- Material:
- Panjang batang:
- Ukuran:
- Kebutuhan komponen:

ASUMSI

1. ...
2. ...

CUTTING LIST

| Batang | Potongan | Total Terpakai | Sisa |
|--------|----------|----------------|------|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |

VALIDASI

- Tidak ada batang melebihi panjang standar.
- Semua sisa >= 0.
- Jumlah potongan sesuai kebutuhan.

TOTAL

- Total batang:
- Total panjang material:
- Total panjang terpakai:
- Total waste:
- Persentase waste:

CATATAN

...

============================================================
PEMERIKSAAN AKHIR
============================================================

SEBELUM mengirim jawaban teknis:

Lakukan pemeriksaan internal:

CHECK 1:
Apakah ada potongan yang lebih panjang dari batang?

CHECK 2:
Apakah jumlah potongan dalam satu batang jika dijumlahkan
melebihi panjang batang?

CHECK 3:
Apakah ada sisa negatif?

CHECK 4:
Apakah jumlah komponen sesuai dengan kebutuhan?

CHECK 5:
Apakah ada komponen yang tiba-tiba dibuat tanpa dasar?

CHECK 6:
Apakah total material sesuai dengan cutting list?

CHECK 7:
Apakah persentase waste dihitung dari data yang benar?

Jika salah satu CHECK gagal:

JANGAN kirim jawaban tersebut.

Perbaiki terlebih dahulu.

============================================================
CODING
============================================================

- berikan kode yang dapat dijalankan
- jangan menghilangkan bagian penting
- pertahankan struktur program pengguna
- jika memperbaiki kode, fokus pada masalah sebenarnya
- gunakan praktik aman
- jangan membocorkan API key
- jangan membocorkan token
- jangan membocorkan secret

============================================================
ROUTER
============================================================

Sistem menggunakan provider AI GRATIS terlebih dahulu.

Jika provider gagal:

- timeout
- rate limit
- 429
- 500
- 502
- 503
- 504
- unavailable
- error
- jawaban kosong

maka otomatis pindah ke provider gratis berikutnya.

Jangan sengaja memakai model berbayar.

============================================================
RAHASIA
============================================================

JANGAN PERNAH menampilkan:

- API key
- token
- password
- secret
- environment variable rahasia
- credential
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

    return memory.setdefault(
        uid,
        []
    )


def remember(
    uid,
    role,
    content,
):

    history(uid).append(
        {
            "role": role,
            "content": content,
        }
    )

    memory[uid] = history(uid)[
        -MAX_MEMORY:
    ]


# ============================================================
# BUILD MESSAGES
# ============================================================

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

Jika pengguna memberikan kode,
analisis kode tersebut sebelum mengubahnya.
""",

        "reasoning": """
TUGAS REASONING.

Analisis masalah secara sistematis.
Periksa kemungkinan penyebab.
Jangan langsung membuat kesimpulan.
""",

        "technical": """
TUGAS TEKNIK DAN MANUFAKTUR.

Ini adalah tugas dengan prioritas AKURASI TINGGI.

Jika ada ukuran material dan kebutuhan potongan:

1. identifikasi semua potongan
2. hitung kebutuhan setiap komponen
3. lakukan packing ke batang fisik
4. pastikan jumlah potongan pada setiap batang
   tidak melebihi panjang batang
5. hitung sisa setiap batang
6. jumlahkan total batang
7. hitung total waste
8. audit kembali seluruh cutting list

ATURAN MUTLAK:

Jika batang = 6m:

4m + 4m TIDAK BOLEH berada pada satu batang.

5m + 2m TIDAK BOLEH berada pada satu batang.

3m + 3m BOLEH.

4m + 2m BOLEH.

5m + 1m BOLEH.

Jika sebuah kebutuhan lebih panjang dari batang,
pecah menjadi beberapa potongan dan lakukan packing
secara benar.

Jangan menciptakan komponen baru tanpa dasar.

Jika data tidak cukup untuk menghitung,
katakan data belum ditentukan.
""",

        "math": """
TUGAS MATEMATIKA.

Hitung secara teliti.
Tampilkan rumus penting.
Tampilkan satuan.
Periksa kembali hasil.
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

    }.get(
        task,
        "",
    )

    return [
        {
            "role": "system",
            "content":
                SYSTEM
                + "\n\n"
                + task_hint,
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
        "cutting",
        "potongan batang",
        "batang 6 meter",
        "batang 6m",
        "batang 12 meter",
        "batang 12m",
        "rangka utama",
        "rangka sekunder",
        "purlin",
        "gording",
        "tiang",
        "balok",
        "bracing",
        "pengaku",
        "base plate",
        "sambungan",
        "atap",
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

    if any(
        x in t
        for x in coding
    ):
        return "coding"

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
# TECHNICAL ANSWER VALIDATOR
# ============================================================

def validate_technical_answer(
    answer,
    original_text,
):

    """
    Pemeriksaan sederhana terhadap kesalahan cutting list
    yang paling berbahaya.

    Validator ini bukan pengganti engineering.
    Tujuannya mencegah kesalahan fisik yang jelas.
    """

    if not answer:
        return True, ""

    text = answer.lower()

    # --------------------------------------------------------
    # Cari pola 4m + 4m pada batang 6m
    # --------------------------------------------------------

    bad_patterns = [

        r"4\s*(?:m|meter)\s*\+\s*4\s*(?:m|meter)"
        r".{0,100}"
        r"(?:6\s*(?:m|meter)|batang\s*6)",

        r"(?:batang\s*6|6\s*(?:m|meter))"
        r".{0,100}"
        r"4\s*(?:m|meter)\s*\+\s*4\s*(?:m|meter)",

    ]

    for pattern in bad_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE |
            re.DOTALL,
        ):

            return (
                False,
                "Terdeteksi cutting list tidak valid: "
                "dua potongan 4m ditempatkan pada "
                "batang 6m."
            )

    # --------------------------------------------------------
    # Pola eksplisit total > 6m
    # --------------------------------------------------------

    invalid_sum_patterns = [

        r"5\s*(?:m|meter)\s*\+\s*2\s*(?:m|meter)",

        r"4\s*(?:m|meter)\s*\+\s*3\s*(?:m|meter)",

        r"3\s*(?:m|meter)\s*\+\s*4\s*(?:m|meter)",

        r"4\s*(?:m|meter)\s*\+\s*4\s*(?:m|meter)",

        r"5\s*(?:m|meter)\s*\+\s*3\s*(?:m|meter)",

    ]

    for pattern in invalid_sum_patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            # Hanya dianggap bermasalah jika konteks
            # menyebut batang 6m.
            if (
                "batang 6" in text
                or "6m" in text
                or "6 m" in text
                or "6 meter" in text
            ):

                return (
                    False,
                    "Terdeteksi kombinasi potongan "
                    "yang melebihi kapasitas batang 6m."
                )

    # --------------------------------------------------------
    # Sisa negatif
    # --------------------------------------------------------

    if re.search(
        r"sisa\s*[:=]\s*-\s*\d",
        text,
        flags=re.IGNORECASE,
    ):

        return (
            False,
            "Terdeteksi sisa material negatif."
        )

    return (
        True,
        ""
    )


# ============================================================
# TECHNICAL CORRECTION PROMPT
# ============================================================

def technical_correction_prompt(
    original_text,
    bad_answer,
    reason,
):

    return f"""
JAWABAN TEKNIS SEBELUMNYA TERDETEKSI SALAH.

ALASAN:
{reason}

PERMINTAAN PENGGUNA:
{original_text}

JAWABAN SEBELUMNYA:
{bad_answer}

ULANGI PERHITUNGAN DARI AWAL.

ATURAN MUTLAK:

1. Setiap batang adalah benda fisik.
2. Jika panjang batang 6m, total semua potongan
   pada batang tersebut harus <= 6m.
3. 4m + 4m TIDAK BOLEH masuk satu batang 6m.
4. 5m + 2m TIDAK BOLEH masuk satu batang 6m.
5. Jangan menciptakan komponen yang tidak diminta.
6. Jangan mengubah kebutuhan pengguna.
7. Jika komponen 8m dibuat dari batang 6m,
   harus dijelaskan bagaimana potongan 8m tersebut
   berasal dari beberapa batang.
8. Hitung jumlah batang berdasarkan cutting list nyata.
9. Hitung sisa setiap batang.
10. Hitung waste dari cutting list final.
11. Audit ulang sebelum menjawab.

Jangan pertahankan jawaban lama jika salah.

Berikan hasil final yang sudah dikoreksi.
"""


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
        r.choices[0]
        .message
        .content
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
            """
Berikan kode yang dapat dijalankan.
Jelaskan perubahan penting.
""",

        "reasoning":
            """
Analisis masalah secara teliti sebelum
memberi kesimpulan.
""",

        "technical":
            """
TUGAS TEKNIK/MANUFAKTUR.

Prioritas utama adalah ketepatan perhitungan.

Jika menghitung cutting list:

- perlakukan setiap batang sebagai benda fisik
- jangan pernah melebihi panjang batang
- 4m + 4m TIDAK boleh pada batang 6m
- 5m + 2m TIDAK boleh pada batang 6m
- hitung setiap batang
- hitung sisa setiap batang
- hitung waste
- jangan mengarang komponen
- jangan mengarang ukuran
- jangan mengarang jumlah

Jika data kurang, katakan data belum ditentukan.

Sebelum menjawab, audit kembali seluruh cutting list.
""",

        "math":
            """
Hitung secara teliti dan tunjukkan asumsi.
""",

        "creative":
            """
Buat hasil kreatif yang siap digunakan.
""",

        "general":
            """
Jawab langsung dan jelas.
""",

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
        r.choices[0]
        .message
        .content
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
# TECHNICAL SECOND PASS
# ============================================================

def verify_and_correct_technical(
    uid,
    original_text,
    answer,
):

    valid, reason = (
        validate_technical_answer(
            answer,
            original_text,
        )
    )

    if valid:

        return (
            answer,
            None,
        )

    log.warning(
        "TECHNICAL VALIDATION FAILED | %s",
        reason,
    )

    correction = (
        technical_correction_prompt(
            original_text,
            answer,
            reason,
        )
    )

    # --------------------------------------------------------
    # Coba Gemini sebagai pemeriksa kedua
    # --------------------------------------------------------

    if gemini:

        try:

            r = gemini.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=correction,
            )

            corrected = (
                r.text
                or ""
            )

            if corrected.strip():

                valid2, reason2 = (
                    validate_technical_answer(
                        corrected,
                        original_text,
                    )
                )

                if valid2:

                    return (
                        corrected,
                        GEMINI_CHAT_MODEL,
                    )

                log.warning(
                    "GEMINI CORRECTION STILL INVALID | %s",
                    reason2,
                )

        except Exception as e:

            log.warning(
                "Gemini technical correction failed: %s",
                str(e)[:300],
            )

    # --------------------------------------------------------
    # Jika koreksi otomatis gagal
    # jangan kirim jawaban yang jelas salah.
    # --------------------------------------------------------

    raise RuntimeError(
        "Jawaban teknis terdeteksi memiliki "
        "ketidaksesuaian pada cutting list. "
        "Sistem menolak mengirim hasil yang "
        "berpotensi salah."
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
    # PROVIDER ORDER
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
    # PROVIDER LOOP
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

            # ------------------------------------------------
            # TECHNICAL VALIDATION
            # ------------------------------------------------

            if task == "technical":

                answer, checker = (
                    verify_and_correct_technical(
                        uid,
                        text,
                        answer,
                    )
                )

                if checker:

                    model = (
                        f"{model} "
                        f"+ checker:{checker}"
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

            log.error(
                "PROVIDER FAILED | "
                "provider=%s | error=%s",
                provider_name,
                error_text[:300],
            )

            # ------------------------------------------------
            # SELALU LANJUT KE PROVIDER BERIKUTNYA
            # ------------------------------------------------

            continue

    # --------------------------------------------------------
    # SEMUA GAGAL
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
# SEND TEXT
# ============================================================

async def send_text(
    chat_id,
    text,
):

    text = (
        text
        or "Tidak ada jawaban."
    )

    for i in range(
        0,
        len(text),
        3900,
    ):

        await tg(
            "sendMessage",
            {
                "chat_id":
                    chat_id,

                "text":
                    text[
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
    # OPENROUTER VISION
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
                    "text":
                        SYSTEM
                        + "\n\n"
                        + prompt,
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
                    model=
                        OPENROUTER_FREE_MODEL,

                    messages=[
                        {
                            "role": "user",
                            "content":
                                content,
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
# VIDEO ANALYSIS
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

Teknik/Manufaktur:
OpenRouter Free → Gemini → Groq

Coding:
OpenRouter Free → Groq → Gemini

Reasoning/Math:
OpenRouter Free → Groq → Gemini

General/Creative:
OpenRouter Free → Gemini → Groq

🔧 Cutting List:
Setiap batang dihitung sebagai batang fisik.
Tidak boleh ada total potongan yang melebihi
panjang batang.

Jika provider gagal → otomatis fallback.

/model → status AI
/reset → hapus memory sesi
/gambar <prompt> → generate gambar gratis
/video → analisis video""",
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
            None,
        )

        await send_text(
            chat_id,
            "✅ Memory sesi dihapus.",
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

Image Generation FREE:
{'Pollinations ✅' if POLLINATIONS_ENABLED and POLLINATIONS_KEY else 'Tidak aktif'}

TECHNICAL:
OpenRouter Free
→ Gemini
→ Groq

CODING:
OpenRouter Free
→ Groq
→ Gemini

REASONING/MATH:
OpenRouter Free
→ Groq
→ Gemini

GENERAL/CREATIVE:
OpenRouter Free
→ Gemini
→ Groq

VISION:
Gemini
→ OpenRouter Free

PAID MODEL ROUTING:
DISABLED

CUTTING LIST VALIDATION:
AKTIF""",
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
                "chat_id":
                    chat_id,

                "action":
                    "typing",
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
            "❌ Request gagal diproses.\n\n"
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

        "free_only":
            True,

        "technical_validator":
            True,

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