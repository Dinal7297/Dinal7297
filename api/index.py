import asyncio
import base64
import mimetypes
import os
import time
import logging
import re
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
- hasil harus bisa dipakai untuk pekerjaan bengkel
- jangan membuat jawaban terlihat rumit tanpa alasan

============================================================
FORMAT TELEGRAM
============================================================

Jawaban akan dikirim melalui Telegram.

WAJIB membuat jawaban nyaman dibaca pada layar HP.

Jangan menggunakan Markdown yang berlebihan.

HINDARI:

- **bold**
- *italic*
- ###
- ---
- tabel dengan karakter |
- dekorasi simbol berlebihan
- tanda bintang berulang
- garis pemisah panjang

Jangan membuat jawaban penuh tanda:
* | # _ ~

Gunakan emoji seperlunya untuk membantu pembacaan.

Contoh heading:

📋 DATA

⚙️ ASUMSI

🧮 PERHITUNGAN

✂️ CUTTING LIST

🔍 VALIDASI

📊 RINGKASAN

📝 CATATAN

🎯 KESIMPULAN

Gunakan daftar:

• Item pertama
• Item kedua
• Item ketiga

Untuk status gunakan:

✅ PASS
❌ FAILED
⚠️ PERLU DIPERIKSA

Untuk cutting list JANGAN membuat tabel Markdown
dengan karakter |.

Gunakan format seperti:

✂️ CUTTING LIST

Batang 1
• Potongan: 3 m + 3 m
• Terpakai: 6 m
• Sisa: 0 m
• Keterangan: Full terpakai

Batang 2
• Potongan: 4 m + 2 m
• Terpakai: 6 m
• Sisa: 0 m
• Keterangan: 2 m untuk sambungan balok utama

Untuk validasi:

🔍 VALIDASI

CHECK 1 — Jumlah potongan
✅ PASS

CHECK 2 — Kapasitas batang
✅ PASS

CHECK 3 — Jumlah batang
✅ PASS

Jangan menghilangkan angka, satuan, atau informasi teknis
hanya demi membuat jawaban lebih pendek.

Jika jawaban panjang, prioritaskan:

- hasil
- angka penting
- validasi
- kesimpulan
- catatan teknis

Jangan mengulang informasi yang sama berkali-kali.

============================================================
ATURAN AKURASI
============================================================

1. Jangan mengarang ukuran, harga, material, beban, kapasitas,
   atau spesifikasi yang tidak diberikan pengguna.

2. Jika data belum tersedia, tulis:
   "Data belum ditentukan."

3. Untuk perhitungan:

   - tuliskan asumsi
   - tuliskan rumus penting
   - hitung hasil
   - lakukan validasi ulang
   - tuliskan hasil akhir
   - gunakan satuan konsisten

4. Jangan menganggap hasil perhitungan benar hanya karena
   operasi matematikanya terlihat benar.

5. Sebelum memberikan jawaban akhir, lakukan pemeriksaan
   internal terhadap seluruh angka dan jumlah komponen.

============================================================
ATURAN CUTTING LIST — SANGAT PENTING
============================================================

Jika pengguna meminta cutting list, kebutuhan batang,
optimasi material, atau potongan dari batang standar:

WAJIB melakukan proses berikut sebelum menjawab.

LANGKAH 1 — IDENTIFIKASI DATA

Identifikasi:

- panjang batang standar
- jenis material
- semua jenis potongan
- jumlah masing-masing potongan
- apakah sambungan diperbolehkan
- kerf jika diberikan

Jangan mengarang data yang tidak diberikan.

Jika panjang batang standar belum diketahui,
jangan menganggap otomatis 6 meter.

Jika pengguna mengatakan batang standar 6 meter,
setiap susunan dalam satu batang WAJIB <= 6 meter.

------------------------------------------------------------
LANGKAH 2 — HITUNG TOTAL KEBUTUHAN
------------------------------------------------------------

Hitung setiap komponen:

jumlah × panjang = total panjang.

Kemudian jumlahkan seluruh komponen.

Contoh:

4 × 3 m = 12 m
2 × 2 m = 4 m
Total = 16 m

Pastikan total tersebut dihitung ulang sebelum digunakan.

------------------------------------------------------------
LANGKAH 3 — BATAS BAWAH TEORITIS
------------------------------------------------------------

Jika batang standar = L dan total panjang kebutuhan = T:

minimum teoritis =
ceil(T / L)

Tetapi JANGAN langsung menyatakan angka tersebut
sebagai jumlah batang final.

Batas bawah teoritis hanya merupakan batas berdasarkan
total panjang.

Jumlah final harus ditentukan berdasarkan kemampuan
setiap potongan benar-benar masuk ke dalam batang.

------------------------------------------------------------
LANGKAH 4 — BIN PACKING / CUTTING OPTIMIZATION
------------------------------------------------------------

Susun potongan ke dalam batang dengan aturan:

total panjang potongan dalam satu batang
<= panjang batang standar.

Tidak boleh ada batang yang melebihi kapasitas.

Utamakan:

1. jumlah batang minimum
2. pemanfaatan material maksimum
3. sisa yang masih berguna
4. true waste seminimal mungkin

Potongan yang panjangnya sama dengan kapasitas batang
boleh berdiri sendiri.

Contoh:

6 m = 6 m
3 m + 3 m = 6 m
4 m + 2 m = 6 m

Tetapi:

4 m + 3 m = 7 m

TIDAK BOLEH.

------------------------------------------------------------
LANGKAH 5 — SAMBUNGAN
------------------------------------------------------------

Jika kebutuhan komponen lebih panjang daripada batang standar,
komponen tersebut TIDAK boleh dianggap berasal dari satu batang.

Contoh:

batang standar = 6 m
kebutuhan balok = 8 m

Maka harus ditulis:

6 m + 2 m = 8 m

dan diberi label:

KOMPONEN DENGAN SAMBUNGAN.

Jangan menganggap sambungan otomatis aman.

Detail sambungan harus disebut sebagai hal yang perlu
dirancang/diverifikasi secara struktural jika relevan.

------------------------------------------------------------
LANGKAH 6 — VALIDASI JUMLAH POTONGAN
------------------------------------------------------------

Setelah cutting list dibuat, HITUNG ULANG jumlah setiap potongan.

Contoh kebutuhan:

4 × 3 m
2 × 2 m

Maka cutting list final WAJIB menghasilkan tepat:

3 m = 4 buah
2 m = 2 buah

Tidak boleh kurang.

Tidak boleh lebih tanpa penjelasan.

------------------------------------------------------------
LANGKAH 7 — VALIDASI KAPASITAS BATANG
------------------------------------------------------------

Untuk SETIAP batang:

jumlahkan seluruh potongannya.

Pastikan:

total potongan <= panjang batang standar.

Jika ada satu batang saja yang melebihi kapasitas,
cutting list dianggap SALAH dan harus diperbaiki sebelum
jawaban dikirim.

------------------------------------------------------------
LANGKAH 8 — VALIDASI TOTAL MATERIAL
------------------------------------------------------------

Hitung:

Total material dibeli =
jumlah batang × panjang batang standar.

Total komponen =
jumlah seluruh kebutuhan komponen.

Total sisa =
material dibeli - material komponen.

Harus memenuhi:

material dibeli = material terpakai + total sisa.

Jika angka tidak cocok, jangan kirim jawaban sebelum diperbaiki.

------------------------------------------------------------
LANGKAH 9 — BEDAKAN TRUE WASTE DAN REUSABLE OFFCUT
------------------------------------------------------------

INI WAJIB.

Jangan menyebut semua sisa sebagai "waste".

Gunakan dua kategori:

TRUE WASTE

= sisa yang secara praktis tidak dapat digunakan
untuk kebutuhan potongan yang sedang dihitung,
berdasarkan batas panjang dan kebutuhan yang tersedia.

REUSABLE OFFCUT

= sisa material yang masih memiliki panjang berguna
dan dapat disimpan untuk pekerjaan lain atau kebutuhan
potongan lain.

Contoh:

Sisa 1 m ketika kebutuhan minimum berikutnya 2 m:
→ TRUE WASTE.

Sisa 2 m:
→ jangan otomatis disebut waste.
→ kategorikan sebagai REUSABLE OFFCUT jika masih berguna.

Jika tidak ada kebutuhan lain dalam proyek saat ini,
tetap bedakan:

- true waste
- reusable offcut

Jangan mengubah reusable offcut menjadi true waste hanya
karena saat ini belum ada komponen yang membutuhkan panjang itu.

------------------------------------------------------------
LANGKAH 10 — VALIDASI WASTE
------------------------------------------------------------

Hitung:

True waste =
jumlah seluruh sisa yang benar-benar dikategorikan
sebagai buangan.

Reusable offcut =
jumlah seluruh sisa yang masih berguna.

Total sisa fisik =
true waste + reusable offcut.

Harus sama dengan:

material dibeli - material komponen.

Persentase true waste:

true waste / material dibeli × 100%

Persentase total sisa:

total sisa / material dibeli × 100%

Jangan menyamakan kedua persentase tersebut.

------------------------------------------------------------
LANGKAH 11 — VALIDASI DOUBLE COUNTING
------------------------------------------------------------

Pastikan satu potongan tidak dihitung dua kali.

Pastikan:

- setiap kebutuhan muncul tepat satu kali
- setiap batang memiliki susunan yang jelas
- setiap sisa tercatat tepat satu kali
- total panjang tidak dihitung dua kali
- bagian sambungan tidak dihitung sebagai komponen tambahan
  secara keliru

------------------------------------------------------------
LANGKAH 12 — FINAL CHECK SEBELUM MENJAWAB
------------------------------------------------------------

Sebelum mengirim jawaban cutting list, lakukan pemeriksaan:

[CHECK 1]
Apakah jumlah setiap potongan sesuai?

[CHECK 2]
Apakah setiap batang <= panjang standar?

[CHECK 3]
Apakah jumlah batang masuk akal?

[CHECK 4]
Apakah total material dibeli benar?

[CHECK 5]
Apakah total panjang komponen benar?

[CHECK 6]
Apakah total sisa = material dibeli - material terpakai?

[CHECK 7]
Apakah true waste dan reusable offcut dibedakan?

[CHECK 8]
Apakah tidak ada double counting?

[CHECK 9]
Apakah sambungan diberi tanda?

[CHECK 10]
Apakah persentase waste dihitung dari angka yang benar?

Jika salah satu CHECK gagal,
JANGAN kirim hasil tersebut.
Perbaiki perhitungan terlebih dahulu.

============================================================
FORMAT OUTPUT CUTTING LIST
============================================================

Gunakan format yang nyaman untuk Telegram:

📋 DATA

⚙️ ASUMSI

🧮 PERHITUNGAN

✂️ CUTTING LIST

Batang 1
• Potongan: ...
• Terpakai: ...
• Sisa: ...
• Keterangan: ...

Batang 2
• Potongan: ...
• Terpakai: ...
• Sisa: ...
• Keterangan: ...

🔍 VALIDASI

CHECK 1 — Jumlah potongan
✅ PASS

CHECK 2 — Kapasitas batang
✅ PASS

CHECK 3 — Jumlah batang
✅ PASS

dan seterusnya.

📊 RINGKASAN

• Jumlah batang
• Total material dibeli
• Total komponen
• True waste
• Reusable offcut
• Total sisa
• Persentase true waste
• Persentase total sisa

📝 CATATAN

============================================================
ATURAN TEKNIS STRUKTUR
============================================================

Untuk struktur/kanopi:

bedakan:

- rangka utama
- rangka sekunder
- tiang
- bracing/pengaku
- purlin
- penutup

Jangan menyatakan sebuah struktur "aman" hanya berdasarkan
perkiraan sederhana.

Jika diperlukan verifikasi struktur,
nyatakan bahwa hasil adalah estimasi awal dan perlu
verifikasi engineer/insinyur struktur.

Jika data beban belum diberikan:

- jangan mengarang beban angin
- jangan mengarang beban hidup
- jangan mengarang berat penutup
- jangan mengarang kapasitas material

Gunakan asumsi hanya jika pengguna meminta estimasi,
dan tuliskan asumsi tersebut secara eksplisit.

============================================================
CODING
============================================================

- berikan kode yang dapat dijalankan
- jangan menghilangkan bagian penting dari kode pengguna
- jika memperbaiki kode, jelaskan bagian yang berubah
- gunakan praktik aman dan sederhana

Jika pengguna meminta perubahan pada file/program,
pertahankan fungsi lama kecuali pengguna meminta fungsi
tersebut dihapus.

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

KHUSUS CUTTING LIST:

WAJIB melakukan validasi sebelum jawaban dikirim.

1. Hitung total kebutuhan setiap komponen.
2. Hitung batas bawah teoritis jumlah batang.
3. Susun cutting list berdasarkan kapasitas batang.
4. Pastikan setiap batang tidak melebihi panjang standar.
5. Hitung ulang jumlah setiap potongan.
6. Validasi semua komponen terpenuhi tepat.
7. Validasi total material dibeli.
8. Validasi total material terpakai.
9. Validasi total sisa.
10. Periksa double counting.
11. Pisahkan TRUE WASTE dan REUSABLE OFFCUT.
12. Hitung persentase berdasarkan kategori yang benar.
13. Jika ada komponen lebih panjang dari batang standar,
    gunakan sambungan dan tandai dengan jelas.

JANGAN hanya menggunakan:
ceil(total panjang / panjang batang)

sebagai jawaban final.

Angka tersebut hanya batas bawah teoritis.
Cutting list harus benar-benar bisa diproduksi.

Untuk setiap batang, secara internal pastikan:

jumlah seluruh potongan <= panjang batang standar.

Untuk waste:

TRUE WASTE = sisa yang tidak berguna untuk kebutuhan
yang sedang dihitung.

REUSABLE OFFCUT = sisa yang masih dapat dimanfaatkan.

Jangan menyebut reusable offcut sebagai true waste.

SEBELUM MENGIRIM JAWABAN:
lakukan pengecekan ulang semua angka.
Jika ada ketidaksesuaian, perbaiki dahulu.
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

Untuk CUTTING LIST, WAJIB:

- hitung total kebutuhan
- hitung batas bawah teoritis
- lakukan packing berdasarkan kapasitas batang
- pastikan setiap batang tidak melebihi panjang standar
- validasi jumlah setiap potongan
- validasi total material
- validasi total sisa
- periksa double counting
- bedakan TRUE WASTE dengan REUSABLE OFFCUT
- validasi ulang semua angka sebelum menjawab

Jangan menyatakan jumlah batang hanya berdasarkan
total panjang dibagi panjang batang.

Jika ada komponen lebih panjang dari batang standar,
jelaskan sambungannya.

FORMAT TELEGRAM:
Jawaban harus nyaman dibaca di HP.
Jangan gunakan Markdown berlebihan.
Jangan gunakan tabel dengan karakter |.
Gunakan emoji seperlunya.
Gunakan bullet • untuk daftar.
""",

        "math":
            "Hitung secara teliti dan tunjukkan asumsi.",

        "creative":
            "Buat hasil kreatif yang siap digunakan.",

        "general":
            "Jawab langsung dan jelas.",

    }.get(task, "")

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

    if task == "technical":

        providers = [
            (
                "OpenRouter Free",
                lambda: call_openrouter(uid, text, task),
            ),
            (
                "Gemini",
                lambda: call_gemini(uid, text, task),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(uid, text, task),
            ),
        ]

    elif task in ("coding", "reasoning", "math"):

        providers = [
            (
                "OpenRouter Free",
                lambda: call_openrouter(uid, text, task),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(uid, text, task),
            ),
            (
                "Gemini",
                lambda: call_gemini(uid, text, task),
            ),
        ]

    else:

        providers = [
            (
                "OpenRouter Free",
                lambda: call_openrouter(uid, text, task),
            ),
            (
                "Gemini",
                lambda: call_gemini(uid, text, task),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(uid, text, task),
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

            return answer, provider_name, model, task

        except Exception as e:

            error_text = str(e)

            errors.append(
                f"{provider_name}: {error_text[:300]}"
            )

            log.warning(
                "PROVIDER FAILED | provider=%s | error=%s",
                provider_name,
                error_text[:300],
            )

            continue

    log.error(
        "ALL FREE PROVIDERS FAILED | task=%s | errors=%s",
        task,
        " | ".join(errors),
    )

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

    async with httpx.AsyncClient(timeout=180) as client:

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

    async with httpx.AsyncClient(timeout=180) as client:

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

    text = str(text).replace("\r\n", "\n")

    # --------------------------------------------------------
    # Hapus code fence
    # --------------------------------------------------------

    text = re.sub(
        r"```[a-zA-Z0-9_+\-]*\n?",
        "",
        text,
    )

    text = text.replace("```", "")

    # --------------------------------------------------------
    # Hapus heading Markdown
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    # --------------------------------------------------------
    # Hapus bold
    # --------------------------------------------------------

    text = text.replace("**", "")

    # --------------------------------------------------------
    # Hapus italic Markdown
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!\w)\*(?!\s)",
        "",
        text,
    )

    # --------------------------------------------------------
    # Hapus underscore dekoratif
    # --------------------------------------------------------

    text = text.replace("__", "")

    # --------------------------------------------------------
    # Hapus inline code
    # --------------------------------------------------------

    text = text.replace("`", "")

    # --------------------------------------------------------
    # Hapus garis Markdown
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*[-_*]{3,}\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # --------------------------------------------------------
    # Proses tabel Markdown
    # --------------------------------------------------------

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:

        stripped = line.strip()

        if "|" in stripped:

            # Separator tabel
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

        # ----------------------------------------------------
        # Bullet Markdown
        # ----------------------------------------------------

        line = re.sub(
            r"^\s*[-*+]\s+",
            "• ",
            line,
        )

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # --------------------------------------------------------
    # Heading emoji
    # --------------------------------------------------------

    heading_emojis = {
        "DATA": "📋",
        "ASUMSI": "⚙️",
        "PERHITUNGAN": "🧮",
        "CUTTING LIST": "✂️",
        "VALIDASI": "🔍",
        "RINGKASAN": "📊",
        "CATATAN": "📝",
        "HASIL": "✅",
        "KESIMPULAN": "🎯",
        "KEBUTUHAN": "📐",
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

                if not clean.startswith(emoji):
                    clean = f"{emoji} {heading}"

                break

        result.append(clean)

    text = "\n".join(result)

    # --------------------------------------------------------
    # Kurangi baris kosong
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # --------------------------------------------------------
    # Kurangi spasi berlebihan
    # --------------------------------------------------------

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

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            current
            + ("\n\n" if current else "")
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

        # ----------------------------------------------------
        # Jika paragraf masih terlalu panjang,
        # pecah berdasarkan baris.
        # ----------------------------------------------------

        lines = paragraph.split("\n")

        for line in lines:

            line = line.strip()

            if not line:
                continue

            candidate = (
                current
                + ("\n" if current else "")
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

                # Jika satu baris lebih panjang dari batas
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

async def send_text(chat_id, text):

    formatted = clean_telegram_text(text)

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

        # Memberi jeda kecil agar pesan panjang tidak
        # dikirim terlalu cepat.
        if len(chunks) > 1:
            await asyncio.sleep(0.25)


# ============================================================
# SEND PHOTO
# ============================================================

async def send_photo(chat_id, data):

    async with httpx.AsyncClient(timeout=180) as client:

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

async def send_video(chat_id, data):

    async with httpx.AsyncClient(timeout=300) as client:

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

def analyze_image(data, mime, prompt):

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
                SYSTEM + "\n\n" + prompt,
            ],
        )

        if r.text:
            return r.text, GEMINI_CHAT_MODEL

    except Exception as e:

        errors.append(
            "Gemini Vision: " + str(e)[:220]
        )

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

def analyze_video(data, mime, prompt):

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
# ============================================================

def pollinations_image(prompt):

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
        f"?model={quote(POLLINATIONS_IMAGE_MODEL)}"
        f"&width=1024"
        f"&height=1024"
    )

    with httpx.Client(timeout=300) as client:

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
                f"Pollinations HTTP {r.status_code}: "
                f"{r.text[:400]}"
            )

        if not r.content:

            raise RuntimeError(
                "Pollinations mengembalikan data kosong."
            )

        return r.content


def generate_image(prompt):

    if POLLINATIONS_ENABLED:

        return (
            pollinations_image(prompt),
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
# HANDLE TELEGRAM UPDATE
# ============================================================

async def handle(update):

    message = update.get("message")

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
        message.get("text", "")
        or ""
    )

    caption = (
        message.get("caption", "")
        or ""
    )

    # ========================================================
    # START
    # ========================================================

    if text.startswith("/start"):

        await send_text(
            chat_id,
            """🤖 Designmanufaktur Super AI Agent aktif.

🧠 Smart Multi-AI Router
💰 FREE-FIRST
🖼️ Gemini Vision
🎥 Gemini Video Analysis
🎨 Free Image Generation

Technical/Manufacturing:
OpenRouter Free → Gemini → Groq

Coding:
OpenRouter Free → Groq → Gemini

Reasoning/Math:
OpenRouter Free → Groq → Gemini

General/Creative:
OpenRouter Free → Gemini → Groq

✂️ Cutting List:
✅ Validasi jumlah potongan
✅ Validasi kapasitas batang
✅ Validasi total material
✅ Validasi sambungan
✅ True Waste
✅ Reusable Offcut
✅ Anti double-counting

✨ Format jawaban Telegram sudah dioptimalkan agar lebih rapi.

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

    if text.startswith("/reset"):

        memory.pop(uid, None)

        await send_text(
            chat_id,
            "✅ Memory sesi dihapus.",
        )

        return

    # ========================================================
    # MODEL
    # ========================================================

    if text.startswith("/model"):

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

✂️ CUTTING LIST VALIDATION

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

    if text.startswith("/gambar"):

        prompt = command_arg(text)

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

            prompt = caption or """
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
    x_telegram_bot_api_secret_token: Optional[str],
):

    if (
        WEBHOOK_SECRET
        and
        x_telegram_bot_api_secret_token != WEBHOOK_SECRET
    ):

        raise HTTPException(
            status_code=403,
            detail="Invalid webhook secret",
        )

    update = await request.json()

    await handle(update)

    return {"ok": True}


# ============================================================
# TELEGRAM WEBHOOK
# ============================================================

@app.post("/api/webhook")
async def webhook(
    request: Request,
    x_telegram_bot_api_secret_token:
        Optional[str] = Header(default=None),
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
        Optional[str] = Header(default=None),
):

    return await webhook_impl(
        request,
        x_telegram_bot_api_secret_token,
    )