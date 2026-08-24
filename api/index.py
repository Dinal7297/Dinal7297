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

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")


# ============================================================
# GEMINI
# ============================================================

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

POLLINATIONS_BASE_URL = "https://gen.pollinations.ai"


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

Jawab dalam Bahasa Indonesia kecuali pengguna meminta bahasa lain.

============================================================
GAYA JAWABAN
============================================================

- langsung ke inti
- praktis
- jelas
- tidak bertele-tele
- gunakan tabel jika membantu
- gunakan satuan yang jelas
- jangan membuat jawaban terlihat rumit tanpa alasan

============================================================
ATURAN AKURASI
============================================================

1. JANGAN mengarang ukuran, harga, material, beban, kapasitas,
   spesifikasi, jumlah komponen, atau dimensi yang tidak diberikan.

2. Jika data belum tersedia, tuliskan:

   "Data belum ditentukan."

3. Untuk perhitungan:
   - tuliskan data
   - tuliskan asumsi
   - gunakan rumus
   - hitung
   - validasi
   - berikan hasil akhir

4. Jangan mengubah data pengguna secara diam-diam.

5. Jika pengguna memberikan angka, gunakan angka tersebut
   secara konsisten.

============================================================
ATURAN KHUSUS CUTTING LIST
============================================================

INI ADALAH ATURAN SANGAT PENTING.

Jika pengguna meminta cutting list, kebutuhan batang,
pemotongan hollow, pipa, besi, atau material panjang:

A. Tentukan panjang batang standar terlebih dahulu.

Contoh:
- batang standar = 6 m

B. Setiap kombinasi dalam satu batang TIDAK BOLEH melebihi
   panjang batang.

Jika batang = 6 m:

3 + 3 = 6     VALID
4 + 2 = 6     VALID
4 + 1 + 1 = 6 VALID
5 + 1 = 6     VALID
3 + 2 = 5     VALID
3 + 3 + 1 = 7 SALAH
4 + 4 = 8     SALAH

C. JANGAN pernah memasukkan potongan 8 m, 7 m, atau panjang
   apa pun yang lebih besar dari batang 6 m sebagai satu potongan
   utuh.

Jika kebutuhan 8 m sedangkan batang hanya 6 m:

- jangan menyebut "8 m tidak muat" lalu berhenti
- jelaskan bahwa diperlukan sambungan jika memang diperbolehkan
- atau pecah menjadi beberapa bagian HANYA jika pengguna
  mengizinkan sambungan
- jangan mengarang detail sambungan struktural

D. Jika TIDAK ADA SAMBUNGAN:

Setiap potongan kebutuhan harus berasal dari satu batang.

E. Jika pengguna memberikan:

4 buah @ 3 m
2 buah @ 2 m

dengan batang 6 m, hasil yang benar adalah:

Batang 1 = 3 + 3 = 6 m
Batang 2 = 3 + 3 = 6 m
Batang 3 = 2 + 2 = 4 m
Sisa = 2 m

Total batang = 3.

F. SELALU hitung batas bawah:

ceil(total kebutuhan / panjang batang)

Tetapi batas bawah tersebut hanya menunjukkan jumlah minimum
secara teoritis.

Setelah itu tetap lakukan PACKING.

G. Untuk cutting list, gunakan prinsip BIN PACKING:

- setiap batang adalah kapasitas tetap
- setiap potongan adalah item
- item tidak boleh melebihi kapasitas
- jumlah semua item harus tepat sesuai kebutuhan
- jumlah batang harus minimum
- setelah jumlah batang minimum tercapai, minimalkan waste
- jangan menggabungkan potongan secara fiktif

H. Setelah membuat cutting list, WAJIB VALIDASI:

1. jumlah setiap potongan sesuai permintaan
2. tidak ada batang melebihi kapasitas
3. tidak ada panjang negatif
4. total terpakai benar
5. total material benar
6. total sisa benar
7. persentase waste benar

I. JANGAN hanya menggunakan:

total panjang / panjang batang

untuk menentukan cutting list.

Harus diperiksa susunan potongannya.

J. Jika kerf/ketebalan potongan tidak diberikan:

nyatakan:

"Kerf/kehilangan akibat potongan belum ditentukan."

Jika pengguna meminta estimasi sederhana, boleh mengabaikan
kerf tetapi harus dinyatakan.

K. Jika sisa material masih dapat digunakan:

bedakan:

- waste teoritis
- sisa material yang masih dapat digunakan

Jangan otomatis menyebut semua sisa sebagai limbah.

============================================================
CONTOH VALIDASI CUTTING LIST
============================================================

DATA:

Batang = 6 m

Kebutuhan:
4 x 3 m
2 x 2 m

Total kebutuhan:

4 x 3 = 12 m
2 x 2 = 4 m

Total = 16 m

Batas bawah:

ceil(16 / 6) = 3 batang

Packing:

Batang 1:
3 + 3 = 6 m
sisa 0 m

Batang 2:
3 + 3 = 6 m
sisa 0 m

Batang 3:
2 + 2 = 4 m
sisa 2 m

Total material:
3 x 6 = 18 m

Terpakai:
16 m

Sisa:
18 - 16 = 2 m

Waste:
2 / 18 x 100 = 11,11%

HASIL:

3 batang.

JANGAN menghasilkan 4 batang.
JANGAN menghasilkan 14 batang.
JANGAN membuat purlin 8 m dari batang 6 m tanpa menjelaskan
sambungan.

============================================================
STRUKTUR / KANOPI
============================================================

Bedakan:

- tiang
- balok utama
- balok sekunder
- purlin
- rangka atap
- bracing/pengaku
- base plate
- plat sambungan
- penutup

Jangan menganggap semua komponen sebagai jenis material
yang sama jika pengguna tidak mengatakan demikian.

Jangan menyatakan struktur "aman" hanya berdasarkan estimasi.

Jika dibutuhkan verifikasi struktur:

"Ini merupakan estimasi awal dan perlu verifikasi engineer/
insinyur struktur sebelum fabrikasi."

============================================================
BEBAN
============================================================

Jika data beban belum diberikan:

JANGAN mengarang:

- beban angin
- beban hidup
- beban mati
- berat penutup
- kapasitas sambungan

Jika pengguna meminta estimasi, nyatakan asumsi dengan jelas.

============================================================
CODING
============================================================

- berikan kode yang dapat dijalankan
- jangan menghilangkan bagian penting kode pengguna
- jika memperbaiki kode, berikan versi lengkap jika diminta
- jangan mengubah API key
- jangan menampilkan secret
- gunakan praktik sederhana dan aman

============================================================
OUTPUT TEKNIS
============================================================

Jika cocok gunakan:

DATA

ASUMSI

PERHITUNGAN

CUTTING LIST

VALIDASI

TOTAL

CATATAN

============================================================
PRINSIP ROUTER
============================================================

Sistem memilih AI berdasarkan jenis tugas.

Prioritas provider GRATIS.

Jangan sengaja menggunakan model berbayar.

Jika provider gagal:

- rate limit
- timeout
- unavailable
- error server
- error koneksi
- jawaban kosong

maka otomatis lanjut ke provider gratis berikutnya.

============================================================
RAHASIA
============================================================

JANGAN PERNAH menampilkan:

- API key
- token
- password
- secret
- environment variable rahasia
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
    return memory.setdefault(uid, [])


def remember(uid, role, content):
    history(uid).append({
        "role": role,
        "content": content,
    })

    memory[uid] = history(uid)[-MAX_MEMORY:]


# ============================================================
# TASK CLASSIFIER
# ============================================================

def classify_task(text):

    t = (text or "").lower()

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
        "index.py",
        "index.php",
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
        "cuttinglist",
        "potongan batang",
        "batang 6 meter",
        "batang 6m",
        "batang standar",
        "rangka utama",
        "rangka sekunder",
        "purlin",
        "bracing",
        "pengaku",
        "balok",
        "tiang",
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

    math = [
        "hitung",
        "perhitungan",
        "berapa",
        "rumus",
        "luas",
        "volume",
        "persentase",
        "matematika",
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

    # Coding paling tinggi
    if any(x in t for x in coding):
        return "coding"

    # Technical sebelum math
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
# MESSAGE BUILDER
# ============================================================

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
Jika pengguna meminta file lengkap,
berikan file lengkap, bukan potongan.
""",

        "reasoning": """
TUGAS REASONING.

Analisis masalah secara sistematis.
Jangan langsung membuat kesimpulan.
Periksa kemungkinan penyebab.
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

ATURAN CUTTING LIST:

Jika batang standar 6 meter:
setiap jumlah potongan dalam satu batang harus <= 6 meter.

Cari kombinasi potongan terbaik.

Contoh:
4 x 3m + 2 x 2m pada batang 6m:

Batang 1 = 3 + 3
Batang 2 = 3 + 3
Batang 3 = 2 + 2

Total = 3 batang.

Jangan membuat potongan 8m dari batang 6m.
Jangan membuat 3+3+1 pada batang 6m.
Jangan menghasilkan jumlah batang berlebihan.

WAJIB validasi ulang:
- jumlah potongan
- kapasitas setiap batang
- total material
- total terpakai
- total sisa
- persentase waste.

Jangan mengarang ukuran yang tidak diberikan.
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
# DETERMINISTIC CUTTING LIST
# ============================================================

def solve_cutting_list(lengths, stock_length=6.0):
    """
    Exact bin-packing sederhana untuk cutting list.
    Tujuan:
    1. jumlah batang minimum
    2. waste minimum

    lengths = list panjang potongan.
    """

    if not lengths:
        return None

    if any(x <= 0 for x in lengths):
        raise ValueError(
            "Panjang potongan harus lebih besar dari 0."
        )

    if any(x > stock_length for x in lengths):
        return None

    # Urutkan dari terbesar ke terkecil
    items = sorted(
        lengths,
        reverse=True,
    )

    best = None

    def search(index, bins):

        nonlocal best

        if index >= len(items):

            count = len(bins)
            waste = sum(
                stock_length - total
                for total, _ in bins
            )

            candidate = (
                count,
                round(waste, 9),
                [
                    (
                        round(total, 6),
                        list(parts),
                    )
                    for total, parts in bins
                ],
            )

            if best is None or (
                candidate[0],
                candidate[1],
            ) < (
                best[0],
                best[1],
            ):
                best = candidate

            return

        item = items[index]

        # Jika jumlah batang sudah tidak mungkin
        # lebih baik dari best, hentikan.
        if best is not None and len(bins) > best[0]:
            return

        tried = set()

        for i in range(len(bins)):

            total, parts = bins[i]

            rounded_total = round(total, 9)

            if rounded_total in tried:
                continue

            tried.add(rounded_total)

            if total + item <= stock_length + 1e-9:

                bins[i] = (
                    total + item,
                    parts + [item],
                )

                search(
                    index + 1,
                    bins,
                )

                bins[i] = (
                    total,
                    parts,
                )

        # Buat batang baru
        bins.append(
            (
                item,
                [item],
            )
        )

        search(
            index + 1,
            bins,
        )

        bins.pop()

    search(0, [])

    return best


def extract_cutting_request(text):
    """
    Mencoba membaca pola sederhana seperti:

    4 buah @ 3 meter
    2 x 2 meter
    4 buah 3m
    2 pcs @ 2m

    Hanya digunakan jika pola cukup jelas.
    """

    t = (text or "").lower()

    # Pastikan konteks cutting
    if not any(
        word in t
        for word in (
            "cutting",
            "potongan",
            "batang",
            "hollow",
            "pipa",
        )
    ):
        return None

    # Cari panjang batang standar
    stock_match = re.search(
        r"(?:batang|panjang).*?"
        r"(\d+(?:[.,]\d+)?)\s*(?:meter|m)\b",
        t,
    )

    stock = 6.0

    if stock_match:
        try:
            stock = float(
                stock_match.group(1).replace(",", ".")
            )
        except Exception:
            stock = 6.0

    pairs = []

    # Pola:
    # 4 buah @ 3 meter
    # 4 pcs x 3m
    # 4 x 3 meter
    pattern = re.compile(
        r"(\d+)\s*"
        r"(?:buah|pcs|pc|potong|potongan)?\s*"
        r"(?:@|x|×)\s*"
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(?:meter|m)\b"
    )

    for match in pattern.finditer(t):

        qty = int(match.group(1))

        length = float(
            match.group(2).replace(",", ".")
        )

        pairs.append(
            (
                qty,
                length,
            )
        )

    if not pairs:
        return None

    # Hilangkan pola yang sebenarnya adalah
    # panjang batang standar, bila tertangkap.
    filtered = []

    for qty, length in pairs:

        if length > stock:
            # Jika pengguna benar-benar meminta
            # potongan > stock, jangan paksa solver.
            return {
                "stock": stock,
                "pairs": pairs,
                "invalid": True,
            }

        filtered.append(
            (
                qty,
                length,
            )
        )

    lengths = []

    for qty, length in filtered:
        lengths.extend(
            [length] * qty
        )

    if not lengths:
        return None

    return {
        "stock": stock,
        "pairs": filtered,
        "lengths": lengths,
        "invalid": False,
    }


def deterministic_cutting_answer(text):
    """
    Jika request cutting list sangat jelas,
    hitung secara deterministik agar AI tidak
    melakukan kesalahan packing.
    """

    data = extract_cutting_request(text)

    if not data:
        return None

    if data.get("invalid"):
        return None

    stock = data["stock"]
    lengths = data["lengths"]
    pairs = data["pairs"]

    result = solve_cutting_list(
        lengths,
        stock,
    )

    if not result:
        return None

    count, waste, bins = result

    total_required = sum(lengths)
    total_stock = count * stock
    total_waste = total_stock - total_required

    # Buat tabel
    lines = []

    lines.append("DATA")
    lines.append("")

    lines.append(f"- Panjang batang standar: {stock:g} meter")

    lines.append("- Kebutuhan potongan:")

    for qty, length in pairs:
        lines.append(
            f"  - {qty} buah @ {length:g} meter"
        )

    lines.append("")
    lines.append("ASUMSI")
    lines.append("")
    lines.append(
        "1. Tidak ada sambungan."
    )
    lines.append(
        "2. Kerf/kehilangan akibat potongan diabaikan."
    )
    lines.append(
        "3. Setiap potongan harus berasal dari satu batang."
    )
    lines.append(
        "4. Tujuan: jumlah batang minimum dan waste minimum."
    )

    lines.append("")
    lines.append("PERHITUNGAN")
    lines.append("")

    lines.append(
        f"Total kebutuhan = {total_required:g} meter"
    )

    lower_bound = int(
        (total_required + stock - 1e-9) // stock
    )

    if total_required % stock > 1e-9:
        lower_bound += 0

    lines.append(
        f"Batas bawah = ceil({total_required:g} / {stock:g}) "
        f"= {lower_bound} batang"
    )

    lines.append("")
    lines.append("CUTTING LIST")
    lines.append("")
    lines.append(
        "| Batang | Potongan | Terpakai | Sisa |"
    )
    lines.append(
        "|---:|---|---:|---:|"
    )

    for idx, (total, parts) in enumerate(bins, 1):

        parts_text = " + ".join(
            f"{p:g}m"
            for p in parts
        )

        remaining = stock - total

        lines.append(
            f"| {idx} | {parts_text} | "
            f"{total:g}m | {remaining:g}m |"
        )

    lines.append("")
    lines.append("VALIDASI")
    lines.append("")

    valid_capacity = all(
        total <= stock + 1e-9
        for total, _ in bins
    )

    valid_count = len(lengths) == sum(
        len(parts)
        for _, parts in bins
    )

    valid_total = abs(
        sum(total for total, _ in bins)
        - total_required
    ) < 1e-9

    lines.append(
        f"- Kapasitas batang: "
        f"{'VALID' if valid_capacity else 'SALAH'}"
    )

    lines.append(
        f"- Jumlah potongan: "
        f"{'VALID' if valid_count else 'SALAH'}"
    )

    lines.append(
        f"- Total panjang terpakai: "
        f"{'VALID' if valid_total else 'SALAH'}"
    )

    lines.append("")
    lines.append("TOTAL")
    lines.append("")

    lines.append(
        f"- Jumlah batang: {count}"
    )

    lines.append(
        f"- Total material: {total_stock:g} meter"
    )

    lines.append(
        f"- Total terpakai: {total_required:g} meter"
    )

    lines.append(
        f"- Total sisa/waste: {total_waste:g} meter"
    )

    waste_percent = (
        total_waste / total_stock * 100
        if total_stock
        else 0
    )

    lines.append(
        f"- Persentase waste: {waste_percent:.2f}%"
    )

    lines.append("")
    lines.append("CATATAN")
    lines.append("")

    lines.append(
        "Hasil dihitung dengan optimasi packing "
        "batang berdasarkan kapasitas batang."
    )

    if total_waste > 0:
        lines.append(
            "Sisa material tidak otomatis berarti limbah; "
            "sisa tersebut masih dapat digunakan untuk "
            "potongan lain jika ukuran dan kebutuhan sesuai."
        )

    return "\n".join(lines)


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
            """
Berikan kode yang dapat dijalankan.
Jika pengguna meminta file lengkap,
berikan file lengkap.
""",

        "reasoning":
            """
Analisis masalah secara teliti sebelum
memberi kesimpulan.
""",

        "technical":
            """
Gunakan pertimbangan teknik/manufaktur.

Untuk cutting list:
- patuhi kapasitas batang
- lakukan bin packing
- jumlah batang harus minimum
- validasi semua potongan
- jangan membuat potongan lebih panjang
  daripada batang
- jangan membuat sambungan kecuali diminta
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

    answer = (
        r.text
        or ""
    )

    if not answer.strip():
        raise RuntimeError(
            "Gemini mengembalikan "
            "jawaban kosong."
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

    # --------------------------------------------------------
    # CUTTING LIST DETERMINISTIC
    # --------------------------------------------------------

    if task == "technical":

        deterministic = (
            deterministic_cutting_answer(
                text
            )
        )

        if deterministic:

            log.info(
                "CUTTING LIST SOLVED "
                "DETERMINISTICALLY"
            )

            return (
                deterministic,
                "Internal Cutting Engine",
                "exact-bin-packing",
                task,
            )

    # --------------------------------------------------------
    # PROVIDER PRIORITY
    # --------------------------------------------------------

    if task == "technical":

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

            log.warning(
                "PROVIDER FAILED | "
                "provider=%s | error=%s",
                provider_name,
                error_text[:300],
            )

            continue

    log.error(
        "ALL FREE PROVIDERS FAILED | "
        "task=%s | errors=%s",
        task,
        " | ".join(errors),
    )

    raise RuntimeError(
        "Semua provider AI GRATIS untuk "
        f"kategori {task} sedang tidak tersedia."
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
        {
            "file_id": file_id
        },
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
# TELEGRAM SEND TEXT
# ============================================================

async def send_text(chat_id, text):

    text = text or "Tidak ada jawaban."

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

async def send_photo(chat_id, data):

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

async def send_video(chat_id, data):

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

def analyze_video(data, mime, prompt):

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

def pollinations_image(prompt):

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
                "Pollinations "
                "mengembalikan data kosong."
            )

        return r.content


def generate_image(prompt):

    if POLLINATIONS_ENABLED:

        return (
            pollinations_image(prompt),
            "Pollinations",
        )

    raise RuntimeError(
        "Generate gambar GRATIS "
        "belum tersedia. Aktifkan "
        "POLLINATIONS_ENABLED=true "
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
        .get(
            "id",
            chat_id,
        )
    )

    text = (
        message.get(
            "text",
            "",
        )
        or ""
    )

    caption = (
        message.get(
            "caption",
            "",
        )
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

Teknik / Cutting List:
→ Exact Cutting Engine
→ OpenRouter Free
→ Gemini
→ Groq

Coding:
→ OpenRouter Free
→ Groq
→ Gemini

Reasoning / Math:
→ OpenRouter Free
→ Groq
→ Gemini

General / Creative:
→ OpenRouter Free
→ Gemini
→ Groq

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

    if text.startswith("/model"):

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

Cutting Engine:
✅ Exact Bin Packing

ROUTING:

Technical / Cutting
→ Internal Cutting Engine
→ OpenRouter Free
→ Gemini
→ Groq

Coding / Reasoning / Math
→ OpenRouter Free
→ Groq
→ Gemini

General / Creative
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
    # GAMBAR
    # ========================================================

    if text.startswith("/gambar"):

        prompt = command_arg(text)

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

    if message.get("video"):

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

            answer = (
                await asyncio.to_thread(
                    analyze_video,
                    data,
                    mime,
                    caption
                    or
                    (
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

    if message.get("photo"):

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
                mimetypes.guess_type(path)[0]
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

        "providers": {
            "gemini":
                bool(gemini),

            "openrouter_free":
                bool(openrouter),

            "groq_free_tier":
                bool(groq),
        },

        "cutting_engine":
            "exact-bin-packing",

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

    await handle(update)

    return {
        "ok": True
    }


# ============================================================
# TELEGRAM WEBHOOK
# ============================================================

@app.post("/api/webhook")
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