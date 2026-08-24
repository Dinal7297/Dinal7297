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
- beton
- pondasi
- lantai
- dinding
- bata
- plester
- acian
- bekisting
- besi tulangan
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
- hasil harus bisa dipakai untuk pekerjaan bengkel dan lapangan
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

Gunakan emoji seperlunya.

Contoh heading:

📋 DATA

⚙️ ASUMSI

🧮 PERHITUNGAN

✂️ CUTTING LIST

📐 KEBUTUHAN SIPIL

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
============================================================

1. Jangan mengarang ukuran, harga, material, beban,
   kapasitas, atau spesifikasi yang tidak diberikan.

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
   internal terhadap seluruh angka.

============================================================
CUTTING LIST
============================================================

Jika pengguna meminta cutting list, kebutuhan batang,
optimasi material, atau potongan dari batang standar:

WAJIB:

1. Identifikasi panjang batang standar.
2. Identifikasi semua potongan.
3. Hitung jumlah × panjang.
4. Hitung total kebutuhan.
5. Hitung batas bawah teoritis.
6. Lakukan bin packing.
7. Pastikan setiap batang tidak melebihi kapasitas.
8. Validasi jumlah potongan.
9. Validasi total material.
10. Validasi total sisa.
11. Bedakan TRUE WASTE dan REUSABLE OFFCUT.
12. Periksa double counting.
13. Jika komponen lebih panjang dari batang standar,
    gunakan sambungan dan tandai.

Jangan otomatis menganggap batang standar 6 meter jika
pengguna tidak menyebutkannya.

TRUE WASTE:
Sisa yang secara praktis tidak dapat digunakan untuk
kebutuhan yang sedang dihitung.

REUSABLE OFFCUT:
Sisa yang masih memiliki panjang berguna dan dapat
disimpan atau digunakan untuk pekerjaan lain.

============================================================
CIVIL CALCULATOR
============================================================

Kamu juga memiliki kemampuan menghitung kebutuhan sipil.

Jenis perhitungan yang didukung:

1. Beton
2. Lantai beton
3. Pondasi beton
4. Sloof / balok beton
5. Kolom beton
6. Dinding bata
7. Plesteran
8. Acian
9. Keramik
10. Bekisting sederhana
11. Galian
12. Besi tulangan
13. Berat besi berdasarkan diameter
14. Kebutuhan batang besi
15. Estimasi semen, pasir, kerikil dan air

Untuk estimasi material beton:

Gunakan asumsi awal yang jelas jika pengguna tidak
memberikan mix design.

Asumsi estimasi default:

- metode campuran nominal 1 : 2 : 3 berdasarkan volume
- faktor volume kering = 1,54
- berat jenis semen untuk estimasi = 1.440 kg/m³
- 1 zak semen = 50 kg
- air menggunakan perkiraan w/c = 0,50

Untuk 1 m³ beton dengan asumsi tersebut:

Semen ≈ 369,6 kg
≈ 7,4 zak semen 50 kg

Pasir ≈ 0,513 m³

Kerikil ≈ 0,770 m³

Air ≈ 185 liter

ANGKA DI ATAS ADALAH ESTIMASI MATERIAL,
BUKAN MIX DESIGN STRUKTURAL.

Untuk pekerjaan struktur penting seperti:

- pondasi bangunan
- kolom
- balok
- sloof
- struktur bertingkat
- struktur menahan beban besar

jangan menyatakan struktur aman hanya dari kalkulator ini.

Jika diperlukan desain struktur, nyatakan bahwa hasil perlu
diverifikasi oleh engineer/insinyur struktur.

============================================================
PERHITUNGAN SIPIL
============================================================

Untuk volume:

Volume = panjang × lebar × tinggi

Untuk luas:

Luas = panjang × lebar

Untuk berat besi:

Berat per meter ≈ diameter² / 162

dalam kg/m untuk diameter dalam mm.

Contoh:

Besi 10 mm:

10² / 162
≈ 0,617 kg/m

Besi 12 mm:

12² / 162
≈ 0,889 kg/m

Untuk kebutuhan batang:

Jumlah batang = ceil(total panjang / panjang batang standar)

Jika batang standar 12 meter, jangan menganggapnya otomatis
jika pengguna tidak menyebutkan panjang batang.

============================================================
ATURAN SIPIL
============================================================

Jika data tidak lengkap:

Jangan mengarang.

Sebutkan data yang diperlukan.

Contoh:

"Untuk menghitung beton saya membutuhkan:
• panjang
• lebar
• tebal"

Jika menggunakan asumsi, tuliskan:

⚙️ ASUMSI

• Mix beton estimasi 1:2:3
• Faktor volume kering 1,54
• Semen 50 kg/zak

Hasil harus dibedakan antara:

ESTIMASI MATERIAL

dan

DESAIN STRUKTURAL.

Kalkulator tidak boleh menyatakan keamanan struktur tanpa
data struktur dan verifikasi yang sesuai.
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

Untuk cutting list lakukan validasi:
1. jumlah potongan
2. kapasitas batang
3. total material
4. total sisa
5. true waste
6. reusable offcut
7. double counting
8. sambungan
""",

        "civil": """
TUGAS PERHITUNGAN SIPIL.

Prioritaskan:
- luas
- volume
- beton
- semen
- pasir
- kerikil
- air
- pondasi
- lantai
- dinding
- bata
- plester
- acian
- keramik
- bekisting
- galian
- besi tulangan

Gunakan satuan konsisten.

Jika data belum lengkap, jangan mengarang.

Jika menggunakan asumsi, tuliskan asumsi.

Bedakan:
ESTIMASI MATERIAL

dengan:

DESAIN STRUKTURAL.

Jangan menyatakan struktur aman hanya dari estimasi.
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
# NUMBER / UNIT HELPERS
# ============================================================

def normalize_number(value):

    if value is None:
        return None

    value = str(value).strip()

    value = value.replace(" ", "")

    if "," in value and "." in value:

        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "")
            value = value.replace(",", ".")

        else:
            value = value.replace(",", "")

    elif "," in value:

        value = value.replace(",", ".")

    try:
        return float(value)

    except Exception:
        return None


def unit_to_meter(value, unit):

    if value is None:
        return None

    unit = (unit or "m").lower()

    if unit in ("mm",):
        return value / 1000

    if unit in ("cm",):
        return value / 100

    if unit in ("dm",):
        return value / 10

    if unit in ("m", "meter", "meters"):
        return value

    return value


def extract_number_unit(text, pattern):

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    value = normalize_number(match.group(1))
    unit = match.group(2) or "m"

    return unit_to_meter(value, unit)


def extract_dimensions(text):

    pattern = (
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(mm|cm|dm|m)?\s*"
        r"(?:x|×|\*)\s*"
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(mm|cm|dm|m)?\s*"
        r"(?:x|×|\*)\s*"
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(mm|cm|dm|m)?"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match:

        a = unit_to_meter(
            normalize_number(match.group(1)),
            match.group(2) or "m",
        )

        b = unit_to_meter(
            normalize_number(match.group(3)),
            match.group(4) or match.group(2) or "m",
        )

        c = unit_to_meter(
            normalize_number(match.group(5)),
            match.group(6) or match.group(4) or match.group(2) or "m",
        )

        return [a, b, c]

    pattern2 = (
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(mm|cm|dm|m)?\s*"
        r"(?:x|×|\*)\s*"
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(mm|cm|dm|m)?"
    )

    match = re.search(
        pattern2,
        text,
        flags=re.IGNORECASE,
    )

    if match:

        a = unit_to_meter(
            normalize_number(match.group(1)),
            match.group(2) or "m",
        )

        b = unit_to_meter(
            normalize_number(match.group(3)),
            match.group(4) or match.group(2) or "m",
        )

        return [a, b]

    return []


def extract_thickness(text):

    patterns = [
        r"tebal(?:nya)?\s*[:=]?\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|dm|m)?",
        r"tebal\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|dm|m)?",
    ]

    for pattern in patterns:

        result = extract_number_unit(
            text,
            pattern,
        )

        if result is not None:
            return result

    return None


def extract_length_after_keyword(text, keywords):

    for keyword in keywords:

        pattern = (
            re.escape(keyword)
            + r"\s*[:=]?\s*"
            r"(\d+(?:[.,]\d+)?)\s*"
            r"(mm|cm|dm|m)?"
        )

        result = extract_number_unit(
            text,
            pattern,
        )

        if result is not None:
            return result

    return None


def extract_area(text):

    result = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(m2|m²|meter persegi)",
        text,
        flags=re.IGNORECASE,
    )

    if result:
        return normalize_number(result.group(1))

    return None


def extract_volume(text):

    result = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(m3|m³|meter kubik)",
        text,
        flags=re.IGNORECASE,
    )

    if result:
        return normalize_number(result.group(1))

    return None


def ceil_int(value):

    return int(math.ceil(value - 1e-12))


def fmt(value, decimals=3):

    if value is None:
        return "-"

    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


# ============================================================
# CONCRETE MATERIAL CALCULATOR
# ============================================================

def concrete_material(volume):

    dry_factor = 1.54

    cement_density = 1440

    cement_ratio = 1 / 6
    sand_ratio = 2 / 6
    gravel_ratio = 3 / 6

    cement_volume = (
        volume
        * dry_factor
        * cement_ratio
    )

    cement_kg = (
        cement_volume
        * cement_density
    )

    cement_bags = cement_kg / 50

    sand = (
        volume
        * dry_factor
        * sand_ratio
    )

    gravel = (
        volume
        * dry_factor
        * gravel_ratio
    )

    water = cement_kg * 0.50

    return {
        "cement_kg": cement_kg,
        "cement_bags": cement_bags,
        "sand_m3": sand,
        "gravel_m3": gravel,
        "water_liter": water,
    }


# ============================================================
# CIVIL CALCULATOR
# ============================================================

def civil_calculator(text):

    original = text or ""

    t = original.lower().strip()

    # --------------------------------------------------------
    # FORCE COMMAND
    # --------------------------------------------------------

    if t.startswith("/sipil"):

        t = command_arg(t)

    # --------------------------------------------------------
    # CONCRETE / FLOOR
    # --------------------------------------------------------

    concrete_keywords = [
        "beton",
        "lantai beton",
        "cor beton",
        "cor lantai",
        "dak beton",
        "cor",
    ]

    if any(k in t for k in concrete_keywords):

        dimensions = extract_dimensions(t)

        thickness = extract_thickness(t)

        volume = extract_volume(t)

        length = None
        width = None

        if len(dimensions) >= 2:

            length = dimensions[0]
            width = dimensions[1]

            if thickness is None and len(dimensions) >= 3:
                thickness = dimensions[2]

        if volume is None:

            if (
                length is not None
                and width is not None
                and thickness is not None
            ):

                volume = (
                    length
                    * width
                    * thickness
                )

        if volume is None:

            return None

        materials = concrete_material(volume)

        title = "📐 KEBUTUHAN BETON"

        if "lantai" in t:
            title = "📐 KEBUTUHAN LANTAI BETON"

        result = [
            title,
            "",
            "📋 DATA",
        ]

        if length is not None:
            result.append(
                f"• Panjang: {fmt(length)} m"
            )

        if width is not None:
            result.append(
                f"• Lebar: {fmt(width)} m"
            )

        if thickness is not None:
            result.append(
                f"• Tebal: {fmt(thickness * 100)} cm"
            )

        result.extend([
            f"• Volume beton: {fmt(volume)} m³",
            "",
            "⚙️ ASUMSI",
            "• Campuran nominal: 1 : 2 : 3",
            "• Faktor volume kering: 1,54",
            "• Semen: 1.440 kg/m³",
            "• 1 zak semen: 50 kg",
            "• Air: w/c sekitar 0,50",
            "",
            "🧮 PERHITUNGAN",
            f"• Semen: {fmt(materials['cement_kg'], 1)} kg",
            f"• Semen: {fmt(materials['cement_bags'], 1)} zak",
            f"• Pasir: {fmt(materials['sand_m3'], 3)} m³",
            f"• Kerikil: {fmt(materials['gravel_m3'], 3)} m³",
            f"• Air: {fmt(materials['water_liter'], 1)} liter",
            "",
            "🔍 VALIDASI",
            "• Volume = panjang × lebar × tebal",
            "• Hasil material merupakan estimasi.",
            "• Untuk struktur penting gunakan mix design/engineering.",
            "",
            "📊 RINGKASAN",
            f"• Beton: {fmt(volume)} m³",
            f"• Semen: sekitar {fmt(materials['cement_bags'], 1)} zak",
            f"• Pasir: sekitar {fmt(materials['sand_m3'], 3)} m³",
            f"• Kerikil: sekitar {fmt(materials['gravel_m3'], 3)} m³",
            "",
            "📝 CATATAN",
            "Estimasi material tidak menggantikan desain struktur."
        ])

        return "\n".join(result)

    # --------------------------------------------------------
    # FOUNDATION / SLOOF / BEAM / COLUMN
    # --------------------------------------------------------

    structural_concrete_keywords = [
        "pondasi beton",
        "pondasi",
        "sloof",
        "balok beton",
        "kolom beton",
        "kolom",
        "balok",
    ]

    if any(k in t for k in structural_concrete_keywords):

        dimensions = extract_dimensions(t)

        volume = extract_volume(t)

        if len(dimensions) >= 3:

            volume = (
                dimensions[0]
                * dimensions[1]
                * dimensions[2]
            )

        if volume is None:

            return None

        materials = concrete_material(volume)

        label = "ELEMEN BETON"

        if "pondasi" in t:
            label = "PONDASI"

        elif "sloof" in t:
            label = "SLOOF"

        elif "kolom" in t:
            label = "KOLOM"

        elif "balok" in t:
            label = "BALOK"

        result = [
            f"📐 KEBUTUHAN {label}",
            "",
            "📋 DATA",
        ]

        if len(dimensions) >= 3:

            result.extend([
                f"• Dimensi: {fmt(dimensions[0])} m × "
                f"{fmt(dimensions[1])} m × "
                f"{fmt(dimensions[2])} m",
            ])

        result.extend([
            f"• Volume: {fmt(volume)} m³",
            "",
            "⚙️ ASUMSI MATERIAL",
            "• Campuran nominal 1 : 2 : 3",
            "• Faktor volume kering 1,54",
            "• Semen 1.440 kg/m³",
            "• Zak semen 50 kg",
            "",
            "🧮 ESTIMASI",
            f"• Semen: {fmt(materials['cement_kg'], 1)} kg",
            f"• Semen: {fmt(materials['cement_bags'], 1)} zak",
            f"• Pasir: {fmt(materials['sand_m3'], 3)} m³",
            f"• Kerikil: {fmt(materials['gravel_m3'], 3)} m³",
            f"• Air: {fmt(materials['water_liter'], 1)} liter",
            "",
            "⚠️ CATATAN PENTING",
            "Ini adalah estimasi volume/material.",
            "Ukuran tulangan, kapasitas dan keamanan struktur",
            "harus dihitung berdasarkan desain struktur."
        ])

        return "\n".join(result)

    # --------------------------------------------------------
    # BRICK WALL
    # --------------------------------------------------------

    brick_keywords = [
        "bata",
        "batu bata",
        "dinding bata",
        "pasang bata",
        "batako",
    ]

    if any(k in t for k in brick_keywords):

        dimensions = extract_dimensions(t)

        area = extract_area(t)

        if len(dimensions) >= 2:

            area = (
                dimensions[0]
                * dimensions[1]
            )

        if area is None:

            return None

        openings = 0

        opening_match = re.search(
            r"bukaan\s*(\d+(?:[.,]\d+)?)\s*(?:m2|m²)",
            t,
            flags=re.IGNORECASE,
        )

        if opening_match:

            openings = normalize_number(
                opening_match.group(1)
            )

        net_area = max(
            0,
            area - openings,
        )

        # Estimasi umum bata merah dengan mortar.
        brick_per_m2 = 60

        brick_count = (
            net_area
            * brick_per_m2
        )

        brick_purchase = ceil_int(
            brick_count
        )

        result = [
            "🧱 KEBUTUHAN DINDING",
            "",
            "📋 DATA",
            f"• Luas dinding kotor: {fmt(area)} m²",
            f"• Bukaan: {fmt(openings)} m²",
            f"• Luas dinding bersih: {fmt(net_area)} m²",
            "",
            "⚙️ ASUMSI",
            "• Estimasi 60 bata/m²",
            "• Angka dapat berubah sesuai ukuran bata",
            "  dan tebal spesi.",
            "",
            "🧮 PERHITUNGAN",
            f"• Bata teoritis: {fmt(brick_count, 1)} buah",
            f"• Pembelian minimum: {brick_purchase} buah",
            "",
            "📊 RINGKASAN",
            f"• Luas: {fmt(net_area)} m²",
            f"• Bata: sekitar {brick_purchase} buah",
            "",
            "📝 CATATAN",
            "Untuk hasil lebih akurat berikan ukuran bata."
        ]

        return "\n".join(result)

    # --------------------------------------------------------
    # PLASTER
    # --------------------------------------------------------

    plaster_keywords = [
        "plester",
        "plesteran",
        "plester dinding",
    ]

    if any(k in t for k in plaster_keywords):

        dimensions = extract_dimensions(t)

        area = extract_area(t)

        if len(dimensions) >= 2:

            area = (
                dimensions[0]
                * dimensions[1]
            )

        if area is None:
            return None

        thickness = extract_thickness(t)

        if thickness is None:
            thickness = 0.015

        wet_volume = (
            area
            * thickness
        )

        dry_volume = (
            wet_volume
            * 1.33
        )

        cement_volume = (
            dry_volume
            * 1 / 5
        )

        sand_volume = (
            dry_volume
            * 4 / 5
        )

        cement_kg = (
            cement_volume
            * 1440
        )

        cement_bags = (
            cement_kg / 50
        )

        result = [
            "🧱 KEBUTUHAN PLESTERAN",
            "",
            "📋 DATA",
            f"• Luas: {fmt(area)} m²",
            f"• Tebal: {fmt(thickness * 100)} cm",
            f"• Volume basah: {fmt(wet_volume, 3)} m³",
            "",
            "⚙️ ASUMSI",
            "• Campuran 1 : 4",
            "• Faktor kering: 1,33",
            "• Semen 1.440 kg/m³",
            "• Zak 50 kg",
            "",
            "🧮 ESTIMASI",
            f"• Semen: {fmt(cement_kg, 1)} kg",
            f"• Semen: {fmt(cement_bags, 1)} zak",
            f"• Pasir: {fmt(sand_volume, 3)} m³",
            "",
            "📝 CATATAN",
            "Kebutuhan aktual dipengaruhi ketebalan dan kondisi dinding."
        ]

        return "\n".join(result)

    # --------------------------------------------------------
    # ACIAN
    # --------------------------------------------------------

    if "acian" in t:

        dimensions = extract_dimensions(t)

        area = extract_area(t)

        if len(dimensions) >= 2:

            area = (
                dimensions[0]
                * dimensions[1]
            )

        if area is None:
            return None

        # Estimasi konsumsi semen acian.
        consumption = 3.0

        cement_kg = (
            area
            * consumption
        )

        cement_bags = (
            cement_kg / 50
        )

        result = [
            "🧱 KEBUTUHAN ACIAN",
            "",
            "📋 DATA",
            f"• Luas: {fmt(area)} m²",
            "",
            "⚙️ ASUMSI",
            "• Konsumsi estimasi: 3 kg semen/m²",
            "",
            "🧮 ESTIMASI",
            f"• Semen: {fmt(cement_kg, 1)} kg",
            f"• Semen: {fmt(cement_bags, 1)} zak",
            "",
            "📝 CATATAN",
            "Konsumsi aktual tergantung ketebalan dan permukaan."
        ]

        return "\n".join(result)

    # --------------------------------------------------------
    # TILE / CERAMIC
    # --------------------------------------------------------

    tile_keywords = [
        "keramik",
        "ubin",
        "lantai keramik",
        "pasang keramik",
    ]

    if any(k in t for k in tile_keywords):

        dimensions = extract_dimensions(t)

        area = extract_area(t)

        if len(dimensions) >= 2:

            area = (
                dimensions[0]
                * dimensions[1]
            )

        if area is None:
            return None

        tile_size_match = re.search(
            r"keramik\s*(\d+)\s*[x×]\s*(\d+)\s*cm",
            t,
            flags=re.IGNORECASE,
        )

        tile_area = None

        if tile_size_match:

            a = normalize_number(
                tile_size_match.group(1)
            )

            b = normalize_number(
                tile_size_match.group(2)
            )

            tile_area = (
                a / 100
                * b / 100
            )

        if tile_area is None:

            tile_area = 0.36

            assumption = (
                "• Ukuran asumsi keramik: 60 × 60 cm"
            )

        else:

            assumption = (
                f"• Ukuran keramik: "
                f"{fmt(math.sqrt(tile_area) * 100)} × "
                f"{fmt(math.sqrt(tile_area) * 100)} cm"
            )

        waste_area = area * 1.05

        tile_count = (
            waste_area
            / tile_area
        )

        result = [
            "⬜ KEBUTUHAN KERAMIK",
            "",
            "📋 DATA",
            f"• Luas lantai: {fmt(area)} m²",
            "",
            "⚙️ ASUMSI",
            assumption,
            "• Cadangan potongan: 5%",
            "",
            "🧮 PERHITUNGAN",
            f"• Luas + cadangan: {fmt(waste_area, 2)} m²",
            f"• Kebutuhan: {ceil_int(tile_count)} keping",
            "",
            "📊 RINGKASAN",
            f"• Keramik: sekitar {ceil_int(tile_count)} keping",
            "",
            "📝 CATATAN",
            "Jika ukuran keramik berbeda, berikan ukurannya."
        ]

        return "\n".join(result)

    # --------------------------------------------------------
    # EXCAVATION / GALIAN
    # --------------------------------------------------------

    excavation_keywords = [
        "galian",
        "penggalian",
        "tanah digali",
        "volume galian",
    ]

    if any(k in t for k in excavation_keywords):

        dimensions = extract_dimensions(t)

        if len(dimensions) >= 3:

            volume = (
                dimensions[0]
                * dimensions[1]
                * dimensions[2]
            )

            result = [
                "⛏️ KEBUTUHAN GALIAN",
                "",
                "📋 DATA",
                f"• Panjang: {fmt(dimensions[0])} m",
                f"• Lebar: {fmt(dimensions[1])} m",
                f"• Kedalaman: {fmt(dimensions[2])} m",
                "",
                "🧮 PERHITUNGAN",
                "Volume = panjang × lebar × kedalaman",
                f"Volume = {fmt(volume)} m³",
                "",
                "📊 RINGKASAN",
                f"• Volume galian: {fmt(volume)} m³",
            ]

            return "\n".join(result)

        return None

    # --------------------------------------------------------
    # REBAR WEIGHT / BARS
    # --------------------------------------------------------

    steel_keywords = [
        "besi tulangan",
        "besi beton",
        "besi ",
        "rebar",
        "tulangan",
    ]

    if any(k in t for k in steel_keywords):

        diameter_match = re.search(
            r"(?:besi|diameter|dia|d)\s*"
            r"(\d+(?:[.,]\d+)?)\s*mm",
            t,
            flags=re.IGNORECASE,
        )

        if not diameter_match:

            diameter_match = re.search(
                r"(\d+(?:[.,]\d+)?)\s*mm",
                t,
                flags=re.IGNORECASE,
            )

        if not diameter_match:
            return None

        diameter = normalize_number(
            diameter_match.group(1)
        )

        weight_per_meter = (
            diameter
            * diameter
            / 162
        )

        length_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*m",
            t,
            flags=re.IGNORECASE,
        )

        total_length = None

        if length_match:

            total_length = normalize_number(
                length_match.group(1)
            )

        quantity_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:batang|btg)",
            t,
            flags=re.IGNORECASE,
        )

        quantity = None

        if quantity_match:

            quantity = ceil_int(
                normalize_number(
                    quantity_match.group(1)
                )
            )

        standard_bar = 12

        standard_match = re.search(
            r"batang\s*(?:standar|std)?\s*"
            r"(\d+(?:[.,]\d+)?)\s*m",
            t,
            flags=re.IGNORECASE,
        )

        if standard_match:

            standard_bar = normalize_number(
                standard_match.group(1)
            )

        if total_length is not None:

            required_bars = ceil_int(
                total_length / standard_bar
            )

            total_steel_length = (
                required_bars
                * standard_bar
            )

            total_weight = (
                total_steel_length
                * weight_per_meter
            )

        elif quantity is not None:

            required_bars = quantity

            total_steel_length = (
                quantity
                * standard_bar
            )

            total_weight = (
                total_steel_length
                * weight_per_meter
            )

        else:

            required_bars = None
            total_steel_length = None
            total_weight = None

        result = [
            "🔩 KALKULATOR BESI",
            "",
            "📋 DATA",
            f"• Diameter: {fmt(diameter)} mm",
            f"• Berat/m: {fmt(weight_per_meter, 3)} kg/m",
            "",
            "🧮 PERHITUNGAN",
            "Rumus berat/m = diameter² ÷ 162",
        ]

        if required_bars is not None:

            result.extend([
                f"• Panjang standar: {fmt(standard_bar)} m",
                f"• Jumlah batang: {required_bars} batang",
                f"• Total panjang dibeli: {fmt(total_steel_length)} m",
                f"• Estimasi berat: {fmt(total_weight, 2)} kg",
            ])

        else:

            result.append(
                "• Data panjang/jumlah batang belum diberikan."
            )

        result.extend([
            "",
            "📝 CATATAN",
            "Berat aktual dapat berbeda sedikit dari tabel pabrik."
        ])

        return "\n".join(result)

    return None


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
    ]

    civil = [
        "sipil",
        "civil",
        "beton",
        "semen",
        "pasir",
        "kerikil",
        "pondasi",
        "pondasi",
        "sloof",
        "kolom beton",
        "balok beton",
        "lantai beton",
        "dinding bata",
        "bata",
        "batako",
        "plester",
        "plesteran",
        "acian",
        "keramik",
        "ubin",
        "bekisting",
        "galian",
        "besi tulangan",
        "tulangan",
        "besi beton",
        "volume galian",
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

    if any(x in t for x in coding):
        return "coding"

    if any(x in t for x in civil):
        return "civil"

    if any(x in t for x in technical):
        return "technical"

    if any(x in t for x in math_keywords):
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

Untuk CUTTING LIST, lakukan validasi:
- total kebutuhan
- batas bawah teoritis
- packing berdasarkan kapasitas
- kapasitas batang
- jumlah potongan
- total material
- total sisa
- true waste
- reusable offcut
- double counting
- sambungan
""",

        "civil":
            """
Gunakan pertimbangan perhitungan sipil yang praktis.

Hitung:
- luas
- volume
- beton
- semen
- pasir
- kerikil
- air
- bata
- plester
- acian
- keramik
- galian
- besi

Jangan mengarang data.

Bedakan estimasi material dengan desain struktur.
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

    elif task in (
        "reasoning",
        "math",
        "civil",
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
        "civil",
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
                for cell
                in stripped.strip("|").split("|")
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

    text = "\n".join(
        cleaned_lines
    )

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

                    line = line[
                        max_length:
                    ]

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

    result = gemini.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=[
            uploaded,
            SYSTEM
            + "\n\n"
            + prompt,
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
📐 Civil Calculator

Technical/Manufacturing:
OpenRouter Free → Gemini → Groq

Coding:
OpenRouter Free → Groq → Gemini

Reasoning/Math/Civil:
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

📐 Civil Calculator:
✅ Beton
✅ Semen
✅ Pasir
✅ Kerikil
✅ Air
✅ Pondasi
✅ Sloof
✅ Kolom
✅ Balok
✅ Lantai
✅ Bata
✅ Plester
✅ Acian
✅ Keramik
✅ Galian
✅ Besi tulangan
✅ Berat besi
✅ Jumlah batang besi

✨ Bisa menggunakan bahasa biasa.

Contoh:

Hitung beton 5 x 10 meter tebal 10 cm

Hitung pondasi 20 x 0,4 x 0,6 meter

Berapa kebutuhan semen pasir kerikil untuk beton 3 m3?

Hitung dinding bata 4 x 3 meter

Hitung berat besi 10 mm sebanyak 20 batang

Perintah:
/model → status AI
/reset → hapus memory sesi
/sipil <perhitungan> → kalkulator sipil
/gambar <prompt> → generate gambar gratis
/video → analisis video

Jika provider AI gagal → otomatis fallback.""",
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

📐 CIVIL CALCULATOR

✅ Beton
✅ Semen
✅ Pasir
✅ Kerikil
✅ Air
✅ Pondasi
✅ Sloof
✅ Kolom
✅ Balok
✅ Bata
✅ Plester
✅ Acian
✅ Keramik
✅ Galian
✅ Besi

🔀 ROUTING

Technical/Manufacturing
→ OpenRouter Free
→ Gemini
→ Groq

Coding/Reasoning/Math/Civil
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

💰 PAID MODEL ROUTING
DISABLED""",
        )

        return

    # ========================================================
    # CIVIL COMMAND
    # ========================================================

    if text.startswith("/sipil"):

        calculation = civil_calculator(
            text
        )

        if calculation:

            await send_text(
                chat_id,
                calculation,
            )

            return

        await send_text(
            chat_id,
            """📐 CIVIL CALCULATOR

Contoh:

/sipil beton 5 x 10 meter tebal 10 cm

/sipil pondasi 20 x 0,4 x 0,6 meter

/sipil beton 3 m3

/sipil dinding bata 4 x 3 meter

/sipil plester 4 x 3 meter

/sipil keramik 5 x 6 meter

/sipil besi 10 mm 20 batang

/sipil besi 12 mm kebutuhan 150 meter""",
        )

        return

    # ========================================================
    # GAMBAR
    # ========================================================

    if text.startswith("/gambar"):

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
                if path.lower().endswith(
                    ".mov"
                )
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
                mimetypes.guess_type(
                    path
                )[0]
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

        # ----------------------------------------------------
        # CIVIL CALCULATOR DETERMINISTIC
        # ----------------------------------------------------

        civil_result = civil_calculator(
            text
        )

        if civil_result:

            remember(
                uid,
                "user",
                text,
            )

            remember(
                uid,
                "assistant",
                civil_result,
            )

            await send_text(
                chat_id,
                civil_result,
            )

            log.info(
                "CIVIL CALCULATOR SUCCESS | text=%s",
                text[:120],
            )

            return

        # ----------------------------------------------------
        # AI CHAT
        # ----------------------------------------------------

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
        "civil_calculator": True,
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