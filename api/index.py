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

============================================================

APP

============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("designmanufaktur")

app = FastAPI(
title="Designmanufaktur Super AI Agent"
)

============================================================

ENVIRONMENT

============================================================

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

============================================================

SYSTEM PROMPT

============================================================

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
- pondasi
- beton
- pembesian
- pasangan bata
- plesteran
- acian
- lantai
- galian
- urugan
- bekisting
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

Jawab dalam Bahasa Indonesia kecuali pengguna meminta bahasa lain.

============================================================
GAYA JAWABAN

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

Jawaban akan dikirim melalui Telegram.

WAJIB membuat jawaban nyaman dibaca pada layar HP.

Jangan menggunakan Markdown yang berlebihan.

HINDARI:

- bold
- italic
- 

---

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

🏗️ STRUKTUR

🔩 MATERIAL

🔍 VALIDASI

📊 RINGKASAN

📝 CATATAN

🎯 KESIMPULAN

Gunakan daftar:

• Item pertama
• Item kedua
• Item ketiga

Untuk status:

✅ PASS
❌ FAILED
⚠️ PERLU DIPERIKSA

============================================================
ATURAN AKURASI

1. Jangan mengarang ukuran, harga, material, beban, kapasitas,
   atau spesifikasi yang tidak diberikan pengguna.

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

4. Jangan menganggap hasil perhitungan benar hanya karena
   operasi matematikanya terlihat benar.

5. Sebelum memberikan jawaban akhir, lakukan pemeriksaan
   internal terhadap seluruh angka.

============================================================
MODUL KEBUTUHAN SIPIL

Jika pengguna meminta:

- kebutuhan sipil
- kebutuhan material bangunan
- hitung beton
- hitung semen
- hitung pasir
- hitung split
- hitung pondasi
- hitung sloof
- hitung kolom
- hitung balok
- hitung lantai
- hitung pasangan bata
- hitung batako
- hitung plester
- hitung acian
- hitung bekisting
- hitung galian
- hitung urugan
- hitung tulangan
- hitung besi beton
- estimasi material proyek

maka gunakan mode PERHITUNGAN SIPIL.

============================================================
JENIS PERHITUNGAN SIPIL

A. VOLUME BETON

Untuk bentuk persegi panjang:

Volume =
panjang × lebar × tinggi

Satuan:
m × m × m = m³

Contoh:

Panjang = 10 m
Lebar = 0,2 m
Tinggi = 0,3 m

Volume =
10 × 0,2 × 0,3
= 0,6 m³

Jika pengguna meminta kebutuhan bahan beton,
jelaskan bahwa kebutuhan semen/pasir/split bergantung
pada mix design atau komposisi yang digunakan.

Jangan mengklaim komposisi tertentu sebagai standar mutlak.

Jika pengguna memberikan komposisi, gunakan komposisi tersebut.

---

B. SEMEN

Jika pengguna memberikan kebutuhan semen per m³:

Semen =
volume beton × kebutuhan semen per m³

Jika menggunakan jumlah zak:

Jumlah zak =
total semen kg / berat per zak

Pembulatan pembelian dilakukan ke atas.

Contoh:

Semen = 360 kg
Zak = 40 kg

360 / 40 = 9 zak

---

C. PASIR DAN SPLIT

Jika pengguna memberikan kebutuhan pasir per m³:

Pasir =
volume beton × kebutuhan pasir per m³

Jika memberikan kebutuhan split:

Split =
volume beton × kebutuhan split per m³

Jangan mengarang koefisien jika pengguna tidak memberikan
komposisi atau metode perhitungan.

---

D. PONDASI

Untuk pondasi menerus:

Volume =
panjang total × luas penampang

Untuk pondasi tapak:

Volume satu pondasi =
panjang × lebar × tinggi

Total =
volume satu pondasi × jumlah pondasi

Untuk pondasi batu kali:

Hitung berdasarkan volume pasangan yang diberikan.

Jangan menentukan dimensi pondasi yang aman tanpa
data tanah, beban, kondisi lokasi, dan perhitungan engineer.

---

E. SLOOF

Volume beton sloof:

panjang × lebar × tinggi

Jika pengguna memberikan ukuran besi:

Hitung kebutuhan tulangan berdasarkan:

- jumlah batang
- diameter
- panjang setiap batang
- panjang standar batang
- sambungan jika ada

Jangan mengarang jumlah tulangan.

---

F. KOLOM

Volume kolom:

luas penampang × tinggi × jumlah kolom

Untuk persegi:

lebar × panjang × tinggi × jumlah

Pembesian dihitung hanya jika data tulangan diberikan.

Jangan menentukan jumlah atau diameter tulangan secara
struktural tanpa dasar engineering.

---

G. BALOK

Volume:

panjang × lebar × tinggi

Jika balok memiliki beberapa segmen,
hitung masing-masing segmen lalu jumlahkan.

---

H. LANTAI / COR PLAT

Volume:

panjang × lebar × tebal

Perhatikan:

tebal harus dikonversi ke meter.

Contoh:

10 cm = 0,10 m

---

I. PASANGAN BATA

Luas dinding:

panjang × tinggi

Jika luas bukaan pintu/jendela tersedia:

Luas bersih =
luas dinding - luas bukaan

Kebutuhan bata:

luas bersih × jumlah bata per m²

Gunakan jumlah bata per m² yang diberikan pengguna.

Jangan mengarang jika tidak diberikan.

Jika pengguna meminta estimasi,
nyatakan asumsi yang digunakan.

---

J. BATAKO

Luas dinding bersih × kebutuhan batako per m².

Tambahkan allowance hanya jika pengguna meminta
atau asumsi tersebut dinyatakan.

---

K. PLESTERAN

Luas plester:

luas permukaan yang diplester

Jika dua sisi dinding:

luas total =
luas satu sisi × 2

Kebutuhan semen/pasir harus menggunakan koefisien
yang diberikan atau asumsi yang dinyatakan.

---

L. ACIAN

Luas acian =
luas bidang yang akan diaci.

Kebutuhan bahan harus berdasarkan koefisien produk
atau data yang diberikan.

---

M. BEKISTING

Hitung luas permukaan yang benar-benar membutuhkan
bekisting.

Jangan menghitung volume sebagai luas bekisting.

Untuk balok:

luas bekisting tergantung sisi yang dibekisting.

Untuk kolom:

luas bekisting =
keliling penampang × tinggi

Untuk pelat:

hitung bidang bawah jika menggunakan bekisting bawah.

---

N. GALIAN

Volume galian:

panjang × lebar × kedalaman

Jika terdapat beberapa galian,
hitung setiap segmen.

---

O. URUGAN

Volume urugan:

volume area yang diisi.

Jika terdapat faktor pemadatan atau faktor susut,
gunakan hanya jika diberikan atau diminta sebagai estimasi.

---

P. PEMBESIAN

Jika pengguna memberikan:

- diameter besi
- jumlah batang
- panjang batang
- panjang kebutuhan
- jarak tulangan
- panjang standar batang

maka hitung kebutuhan secara rinci.

Berat besi dapat dihitung dengan pendekatan:

Berat per meter =
diameter² / 162

dengan diameter dalam mm.

Contoh:

D10:

10² / 162
= 0,617 kg/m

Total berat:

panjang total × berat per meter.

Hasil ini merupakan pendekatan berat teoritis.

---

Q. WASTE MATERIAL SIPIL

Jika pengguna meminta estimasi pembelian material,
bedakan:

• kebutuhan teoritis
• kebutuhan pembelian
• allowance/waste
• sisa material

Jangan menyebut semua selisih sebagai waste.

Jika allowance diberikan:

Kebutuhan pembelian =
kebutuhan teoritis × (1 + allowance)

Contoh allowance 5%:

kebutuhan × 1,05

Jika pengguna tidak memberikan allowance,
jangan mengarang angka sebagai fakta.

---

R. VALIDASI SIPIL

Setiap perhitungan sipil harus memeriksa:

CHECK 1 — Satuan
CHECK 2 — Dimensi
CHECK 3 — Volume
CHECK 4 — Jumlah komponen
CHECK 5 — Konversi satuan
CHECK 6 — Material
CHECK 7 — Pembulatan pembelian
CHECK 8 — Double counting
CHECK 9 — Waste/allowance
CHECK 10 — Konsistensi total

Jika salah satu gagal:
perbaiki sebelum menjawab.

============================================================
BATASAN ENGINEERING SIPIL

AI boleh membantu:

- estimasi volume
- estimasi kebutuhan material
- perhitungan matematika
- rekap pekerjaan
- quantity takeoff
- cutting list
- estimasi awal
- pengecekan angka

Tetapi AI tidak boleh menyatakan struktur pasti aman
tanpa data dan analisis engineering yang memadai.

Untuk:

- pondasi
- kolom
- balok
- pelat
- struktur baja
- struktur beton
- tulangan
- kapasitas tanah
- beban gempa
- beban angin
- kapasitas sambungan

jika data tidak lengkap, nyatakan:

"Ini merupakan estimasi/perhitungan awal dan bukan
pengganti desain atau verifikasi engineer struktur."

============================================================
CUTTING LIST

Jika pengguna meminta cutting list:

WAJIB melakukan:

1. Identifikasi panjang batang standar.
2. Identifikasi semua potongan.
3. Hitung jumlah masing-masing.
4. Hitung total kebutuhan.
5. Hitung batas bawah teoritis.
6. Lakukan packing potongan.
7. Pastikan setiap batang <= panjang standar.
8. Validasi jumlah setiap potongan.
9. Validasi total material.
10. Validasi total sisa.
11. Pisahkan TRUE WASTE dan REUSABLE OFFCUT.
12. Periksa double counting.
13. Jika komponen lebih panjang dari batang standar,
    gunakan sambungan dan tandai.

Jangan menggunakan:

ceil(total panjang / panjang batang)

sebagai jawaban final.

Angka tersebut hanya batas bawah teoritis.

============================================================
FORMAT CUTTING LIST

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

============================================================
TRUE WASTE

TRUE WASTE:

Sisa yang secara praktis tidak dapat digunakan untuk
kebutuhan yang sedang dihitung.

============================================================
REUSABLE OFFCUT

REUSABLE OFFCUT:

Sisa material yang masih memiliki panjang berguna dan
dapat digunakan untuk pekerjaan lain atau komponen lain.

Jangan mengubah reusable offcut menjadi true waste.

============================================================
CODING

- berikan kode yang dapat dijalankan
- jangan menghilangkan bagian penting dari kode pengguna
- jika memperbaiki kode, pertahankan fungsi lama
- gunakan praktik aman dan sederhana

============================================================
PRIVASI

JANGAN PERNAH:

- menampilkan API key
- menampilkan token
- menampilkan password
- menampilkan secret
- membocorkan rahasia sistem
  """

============================================================

AI CLIENTS

============================================================

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

============================================================

MEMORY

============================================================

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
WAJIB validasi seluruh angka sebelum menjawab.
""",

    "civil": """

TUGAS KEBUTUHAN SIPIL.

Prioritaskan:

- quantity takeoff
- volume pekerjaan
- beton
- semen
- pasir
- split
- pondasi
- sloof
- kolom
- balok
- plat/lantai
- pasangan bata/batako
- plester
- acian
- bekisting
- galian
- urugan
- pembesian
- berat besi
- kebutuhan material

WAJIB:

1. Identifikasi data.
2. Gunakan satuan konsisten.
3. Tulis rumus penting.
4. Hitung setiap item.
5. Jumlahkan total.
6. Validasi ulang.
7. Bedakan kebutuhan teoritis dengan kebutuhan pembelian.
8. Jangan mengarang koefisien material.
9. Jangan mengarang ukuran struktur.
10. Jika data engineering tidak cukup, nyatakan sebagai
    estimasi awal.

Untuk beton:

volume = panjang × lebar × tinggi

Untuk besi:

berat per meter ≈ diameter² / 162

Jika diameter dalam mm.

Untuk pasangan:

luas = panjang × tinggi

Kurangi bukaan jika data tersedia.

Untuk galian:

volume = panjang × lebar × kedalaman

Untuk bekisting:

hitung luas permukaan yang benar-benar dibekisting,
bukan volume.

Selalu lakukan validasi satuan dan angka.
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

============================================================

TASK CLASSIFIER

============================================================

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
    "sipil",
    "pekerjaan sipil",
    "kebutuhan sipil",
    "material bangunan",
    "bahan bangunan",
    "beton",
    "semen",
    "pasir",
    "split",
    "kerikil",
    "cor",
    "pondasi",
    "pondasi batu kali",
    "pondasi tapak",
    "footplat",
    "sloof",
    "kolom beton",
    "balok beton",
    "plat beton",
    "pelat beton",
    "lantai beton",
    "bata",
    "batu bata",
    "batako",
    "hebel",
    "dinding",
    "pasangan",
    "pasangan bata",
    "plester",
    "plesteran",
    "acian",
    "bekisting",
    "galian",
    "galian tanah",
    "urugan",
    "uruk",
    "tulangan",
    "tulangan beton",
    "besi beton",
    "besi tulangan",
    "begel",
    "sengkang",
    "wiremesh",
    "volume beton",
    "volume pondasi",
    "volume galian",
    "quantity takeoff",
    "rencana anggaran",
    "rab",
    "kebutuhan semen",
    "kebutuhan pasir",
    "kebutuhan split",
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

============================================================

OPENROUTER

============================================================

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

============================================================

GEMINI

============================================================

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

Untuk CUTTING LIST:

- hitung total kebutuhan
- hitung batas bawah teoritis
- lakukan packing berdasarkan kapasitas batang
- pastikan setiap batang tidak melebihi panjang standar
- validasi jumlah setiap potongan
- validasi total material
- validasi total sisa
- periksa double counting
- bedakan TRUE WASTE dan REUSABLE OFFCUT
- validasi ulang semua angka

Jangan hanya menggunakan total panjang dibagi panjang batang.
""",

    "civil":
        """

Gunakan mode perhitungan kebutuhan sipil.

Prioritaskan:

- volume pekerjaan
- beton
- semen
- pasir
- split
- pondasi
- sloof
- kolom
- balok
- plat/lantai
- bata/batako
- plester
- acian
- bekisting
- galian
- urugan
- pembesian
- berat besi
- quantity takeoff

WAJIB:

1. Tampilkan data.
2. Tampilkan asumsi.
3. Gunakan satuan konsisten.
4. Tampilkan rumus penting.
5. Hitung hasil.
6. Validasi ulang.
7. Bedakan kebutuhan teoritis dan kebutuhan pembelian.
8. Jangan mengarang koefisien.
9. Jangan mengarang dimensi struktur.
10. Jika data engineering belum lengkap, sebutkan bahwa
    hasil adalah estimasi awal.

Rumus dasar:

Volume beton =
panjang × lebar × tinggi

Luas dinding =
panjang × tinggi

Volume galian =
panjang × lebar × kedalaman

Berat besi teoritis =
diameter² / 162 × panjang

Diameter dalam mm.

Untuk kebutuhan semen, pasir, split:
gunakan koefisien yang diberikan pengguna.
Jika tidak tersedia, jangan menyajikan angka koefisien
sebagai fakta pasti.

FORMAT TELEGRAM:
Jangan gunakan tabel Markdown dengan karakter |.
Gunakan bullet •.
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

============================================================

GROQ

============================================================

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

============================================================

SMART CHAT ROUTER

============================================================

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

============================================================

TELEGRAM API

============================================================

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

============================================================

TELEGRAM FILE

============================================================

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

============================================================

TELEGRAM RESPONSE FORMATTER

============================================================

def clean_telegram_text(text):

if not text:
    return "Tidak ada jawaban."

text = str(text).replace("\r\n", "\n")

text = re.sub(
    r"```[a-zA-Z0-9_+\-]*\n?",
    "",
    text,
)

text = text.replace("```", "")

text = re.sub(
    r"^\s*#{1,6}\s*",
    "",
    text,
    flags=re.MULTILINE,
)

text = text.replace("**", "")

text = re.sub(
    r"(?<!\w)\*(?!\s)",
    "",
    text,
)

text = text.replace("__", "")
text = text.replace("`", "")

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
    "CUTTING LIST": "✂️",
    "VALIDASI": "🔍",
    "RINGKASAN": "📊",
    "CATATAN": "📝",
    "HASIL": "✅",
    "KESIMPULAN": "🎯",
    "KEBUTUHAN": "📐",
    "KEBUTUHAN SIPIL": "📐",
    "MATERIAL": "🔩",
    "SAMBUNGAN": "🔧",
    "WASTE": "♻️",
    "TRUE WASTE": "🗑️",
    "REUSABLE OFFCUT": "♻️",
    "BETON": "🏗️",
    "PEMBESIAN": "🔩",
    "PONDASI": "🧱",
    "GALIAN": "⛏️",
    "URUGAN": "🪨",
    "BEKISTING": "🪚",
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

============================================================

SMART TELEGRAM CHUNK

============================================================

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

============================================================

SEND TEXT

============================================================

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

    if len(chunks) > 1:
        await asyncio.sleep(0.25)

============================================================

SEND PHOTO

============================================================

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

============================================================

SEND VIDEO

============================================================

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

============================================================

GEMINI VISION

============================================================

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

============================================================

GEMINI VIDEO

============================================================

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

============================================================

IMAGE GENERATION

============================================================

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

============================================================

COMMAND ARGUMENT

============================================================

def command_arg(text):

parts = text.split(
    maxsplit=1
)

return (
    parts[1].strip()
    if len(parts) > 1
    else ""
)

============================================================

HANDLE TELEGRAM UPDATE

============================================================

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

🔧 Technical/Manufacturing:
OpenRouter Free → Gemini → Groq

📐 Civil:
OpenRouter Free → Gemini → Groq

💻 Coding:
OpenRouter Free → Groq → Gemini

🧠 Reasoning/Math:
OpenRouter Free → Groq → Gemini

🎨 General/Creative:
OpenRouter Free → Gemini → Groq

✂️ Cutting List:
✅ Validasi jumlah potongan
✅ Validasi kapasitas batang
✅ Validasi total material
✅ Validasi sambungan
✅ True Waste
✅ Reusable Offcut
✅ Anti double-counting

📐 Kebutuhan Sipil:
✅ Volume beton
✅ Semen
✅ Pasir
✅ Split
✅ Pondasi
✅ Sloof
✅ Kolom
✅ Balok
✅ Plat/lantai
✅ Bata/batako
✅ Plester
✅ Acian
✅ Bekisting
✅ Galian
✅ Urugan
✅ Pembesian
✅ Berat besi
✅ Quantity Takeoff

✨ Format jawaban Telegram sudah dioptimalkan.

Jika provider gagal → otomatis fallback.

/model → status AI
/reset → hapus memory sesi
/gambar <prompt> → generate gambar
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
✅ Beton
✅ Pondasi
✅ Pembesian
✅ Galian
✅ Urugan
✅ Bekisting
✅ Quantity Takeoff

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

============================================================

ROOT

============================================================

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
    "features": {
        "technical": True,
        "civil": True,
        "cutting_list": True,
        "quantity_takeoff": True,
        "vision": True,
        "video": True,
        "image_generation": POLLINATIONS_ENABLED,
    },
    "models": {
        "gemini": GEMINI_CHAT_MODEL,
        "openrouter": OPENROUTER_FREE_MODEL,
        "groq_coding": GROQ_CODING_MODEL,
        "groq_reasoning": GROQ_REASONING_MODEL,
        "groq_fast": GROQ_FAST_MODEL,
    },
}

============================================================

API

============================================================

@app.get("/api")
async def api_root():

return await root()

============================================================

WEBHOOK IMPLEMENTATION

============================================================

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

============================================================

TELEGRAM WEBHOOK

============================================================

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

============================================================

LEGACY ROOT WEBHOOK

============================================================

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