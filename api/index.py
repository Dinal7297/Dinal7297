import asyncio
import base64
import mimetypes
import os
import time
import logging
import re
import math
import json
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

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    ""
)

CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    ""
)

CLOUDFLARE_IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
)


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
- estimasi material
- engineering
- perhitungan teknis
- coding
- bisnis
- konten
- pemasaran

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
- gunakan satuan yang jelas
- hasil harus bisa dipakai untuk pekerjaan lapangan
- jangan membuat jawaban terlihat rumit tanpa alasan

============================================================
FORMAT TELEGRAM
============================================================

Jangan menggunakan Markdown berlebihan.

HINDARI:

**bold**
*italic*
###
---
tabel dengan karakter |

Gunakan heading:

📋 DATA
⚙️ ASUMSI
🧮 PERHITUNGAN
✂️ CUTTING LIST
🏗️ CIVIL CALCULATOR
🔩 MATERIAL
🔍 VALIDASI
📊 RINGKASAN
📝 CATATAN
🎯 KESIMPULAN

Gunakan:

• Item pertama
• Item kedua
• Item ketiga

Status:

✅ PASS
❌ FAILED
⚠️ PERLU DIPERIKSA

============================================================
ATURAN AKURASI
============================================================

1. Jangan mengarang ukuran, harga, material, beban,
   kapasitas, mutu beton, atau spesifikasi yang tidak diberikan.

2. Jika data belum tersedia, tulis:
   "Data belum ditentukan."

3. Untuk perhitungan:
   - tuliskan asumsi
   - tuliskan rumus penting
   - hitung hasil
   - validasi ulang
   - tuliskan hasil akhir
   - gunakan satuan konsisten

4. Jangan menyatakan struktur aman hanya berdasarkan
   perkiraan sederhana.

5. Untuk struktur yang memerlukan desain:
   hasil adalah estimasi awal dan harus diverifikasi
   engineer/insinyur struktur.

============================================================
CIVIL CALCULATOR
============================================================

Bot memiliki kemampuan menghitung:

1. Volume beton
2. Kebutuhan semen beton
3. Kebutuhan pasir beton
4. Kebutuhan kerikil beton
5. Kebutuhan air
6. Sloof
7. Kolom
8. Balok
9. Plat beton
10. Footplat
11. Pondasi beton
12. Pondasi batu kali
13. Dinding bata
14. Dinding batako
15. Plesteran
16. Acian
17. Lantai beton
18. Screed
19. Galian
20. Urugan
21. Besi tulangan
22. Berat besi
23. Jumlah batang besi standar
24. Kebutuhan kawat bendrat secara estimasi

============================================================
ASUMSI CIVIL
============================================================

Jika pengguna meminta estimasi material beton dan tidak
memberikan mix design, gunakan asumsi nominal 1 : 2 : 3.

Faktor volume kering beton:
1,54

Estimasi berat semen:
1.440 kg/m³

Berat 1 zak semen:
50 kg

Untuk estimasi beton 1 : 2 : 3:

Total bagian = 6

Semen:
1/6 × volume kering

Pasir:
2/6 × volume kering

Kerikil:
3/6 × volume kering

Air:
gunakan w/c sekitar 0,50 dari berat semen,
hanya sebagai estimasi awal.

HASIL INI BUKAN MIX DESIGN LABORATORIUM.

============================================================
RUMUS BESI
============================================================

Berat besi per meter:

berat kg/m =
diameter² / 162

Contoh:
D10:
10² / 162 = 0,617 kg/m

D13:
13² / 162 = 1,043 kg/m

D16:
16² / 162 = 1,580 kg/m

Jika panjang total besi diketahui:

berat total =
panjang total × berat per meter

Batang standar default:
12 meter

Jika pengguna memberikan panjang batang berbeda,
gunakan panjang tersebut.

Jangan menganggap jumlah besi struktur aman hanya
berdasarkan hitungan berat.

============================================================
ATURAN STRUKTUR
============================================================

Untuk:

- sloof
- kolom
- balok
- plat
- footplat
- pondasi

pisahkan:

volume beton
estimasi material
besi jika data tersedia
asumsi
validasi
catatan struktural

Jika diameter, jumlah tulangan, jarak begel,
mutu beton, beban, kondisi tanah, bentang,
atau data struktural tidak diberikan:

jangan mengarang.

Tulis:
"Data struktur belum ditentukan."

============================================================
CUTTING LIST
============================================================

Jika pengguna meminta cutting list:

1. Hitung semua kebutuhan.
2. Hitung batas bawah teoritis.
3. Lakukan packing.
4. Setiap batang tidak boleh melebihi kapasitas.
5. Validasi jumlah potongan.
6. Validasi jumlah batang.
7. Validasi total material.
8. Validasi total sisa.
9. Bedakan TRUE WASTE dan REUSABLE OFFCUT.
10. Periksa double counting.
11. Jika komponen lebih panjang dari batang standar,
    gunakan sambungan dan tandai.

Jangan menggunakan ceil(total / panjang batang)
sebagai hasil final.

Angka tersebut hanya batas bawah teoritis.

============================================================
PRIVASI
============================================================

JANGAN PERNAH menampilkan:

- API key
- token
- password
- secret
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

# ============================================================
# PERSISTENT MEMORY — SEPARATE GITHUB REPOSITORY
# ============================================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Dinal7297/designmanufaktur-memory")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_MEMORY_DIR = "memory"
GITHUB_API = "https://api.github.com"


def history(uid):
    return memory.setdefault(uid, [])


def remember(uid, role, content):

    history(uid).append({
        "role": role,
        "content": content,
    })

    memory[uid] = history(uid)[-MAX_MEMORY:]


def _memory_path(uid):
    return f"{GITHUB_MEMORY_DIR}/{str(uid)}.json"


async def load_persistent_memory(uid):
    uid = str(uid)
    if not GITHUB_TOKEN or not GITHUB_REPO:
        memory.setdefault(uid, [])
        return
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{_memory_path(uid)}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
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
            saved = saved.get("memory", [])
        if not isinstance(saved, list):
            saved = []
        memory[uid] = saved[-MAX_MEMORY:]
        log.info("PERSISTENT MEMORY LOAD OK | uid=%s | items=%s", uid, len(memory[uid]))
    except Exception as e:
        log.warning("PERSISTENT MEMORY LOAD FAILED | uid=%s | %s", uid, str(e)[:300])
        memory.setdefault(uid, [])


async def save_persistent_memory(uid):
    uid = str(uid)
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.warning("PERSISTENT MEMORY SAVE SKIPPED | GITHUB_TOKEN/GITHUB_REPO belum tersedia")
        return
    raw = json.dumps({"user_id": uid, "memory": history(uid)[-MAX_MEMORY:], "updated_at": int(time.time())}, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{_memory_path(uid)}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            current = await client.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
            body = {"message": f"memory: update {uid}", "content": encoded, "branch": GITHUB_BRANCH}
            if current.status_code == 200:
                body["sha"] = current.json().get("sha")
            elif current.status_code != 404:
                current.raise_for_status()
            saved = await client.put(url, headers=headers, json=body)
        if saved.status_code not in (200, 201):
            saved.raise_for_status()
        log.info("PERSISTENT MEMORY SAVE OK | uid=%s", uid)
    except Exception as e:
        log.warning("PERSISTENT MEMORY SAVE FAILED | uid=%s | %s", uid, str(e)[:300])


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
- pekerjaan sipil

Untuk cutting list:
WAJIB validasi seluruh angka sebelum menjawab.
""",

        "civil": """
TUGAS CIVIL CALCULATOR.

Prioritaskan:
- volume
- dimensi
- kebutuhan material
- semen
- pasir
- kerikil
- air
- besi
- berat besi
- pondasi
- dinding
- plester
- acian
- galian
- urugan

Jika data tidak diberikan:
jangan mengarang.

Untuk struktur:
hasil adalah estimasi awal,
bukan pengganti desain engineer.
""",

        "math": """
TUGAS MATEMATIKA.

Hitung dengan teliti.
Tampilkan rumus penting.
Gunakan satuan.
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
# NUMBER PARSER
# ============================================================

def parse_number(value):
    """
    Mengubah angka seperti:
    5
    5.5
    5,5
    1.000
    1,000
    menjadi float.
    """

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


def fmt(value, decimals=3):

    if value is None:
        return "-"

    if abs(value - round(value)) < 0.000001:
        return str(int(round(value)))

    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


# ============================================================
# DIMENSION PARSER
# ============================================================

def extract_dimensions(text):

    t = text.lower()

    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)",
    ]

    for pattern in patterns:

        m = re.search(pattern, t)

        if m:

            values = [
                parse_number(x)
                for x in m.groups()
            ]

            return values

    return []


def extract_value_with_unit(text, units):

    unit_pattern = "|".join(
        re.escape(x)
        for x in units
    )

    pattern = (
        r"(\d+(?:[.,]\d+)?)"
        r"\s*("
        + unit_pattern
        + r")"
    )

    m = re.search(
        pattern,
        text.lower()
    )

    if not m:
        return None

    value = parse_number(m.group(1))
    unit = m.group(2)

    if value is None:
        return None

    return value, unit


def to_meter(value, unit):

    unit = unit.lower()

    if unit in ("mm",):
        return value / 1000

    if unit in ("cm",):
        return value / 100

    if unit in ("m", "meter", "meters"):
        return value

    return value


# ============================================================
# CIVIL CONCRETE CALCULATOR
# ============================================================

def concrete_materials(
    volume,
    mix=(1, 2, 3),
    dry_factor=1.54,
    cement_density=1440,
    sack_weight=50,
    wc=0.50,
):

    a, b, c = mix

    total_parts = a + b + c

    dry_volume = volume * dry_factor

    cement_volume = (
        dry_volume * a / total_parts
    )

    sand_volume = (
        dry_volume * b / total_parts
    )

    gravel_volume = (
        dry_volume * c / total_parts
    )

    cement_kg = (
        cement_volume * cement_density
    )

    cement_sacks = math.ceil(
        cement_kg / sack_weight
    )

    water_liter = cement_kg * wc

    return {
        "volume": volume,
        "dry_volume": dry_volume,
        "cement_kg": cement_kg,
        "cement_sacks": cement_sacks,
        "sand_m3": sand_volume,
        "gravel_m3": gravel_volume,
        "water_liter": water_liter,
    }


def civil_concrete_calculation(
    text,
    title="KEBUTUHAN BETON"
):

    dims = extract_dimensions(text)

    if len(dims) == 3:

        p, l, t = dims

    else:

        return None

    # Jika ukuran terlihat seperti cm/mm,
    # pengguna sebaiknya menyebut unit.
    # Default hanya meter jika tidak ada unit.

    volume = p * l * t

    result = concrete_materials(
        volume
    )

    answer = f"""
🏗️ {title}

📋 DATA

• Panjang: {fmt(p)} m
• Lebar: {fmt(l)} m
• Tebal/Tinggi: {fmt(t)} m
• Volume beton: {fmt(volume)} m³

⚙️ ASUMSI

• Campuran nominal: 1 : 2 : 3
• Faktor volume kering: 1,54
• Berat semen: 1.440 kg/m³
• 1 zak semen: 50 kg
• Air: w/c sekitar 0,50

🧮 PERHITUNGAN

• Semen: sekitar {fmt(result["cement_kg"])} kg
• Semen: sekitar {result["cement_sacks"]} zak
• Pasir: sekitar {fmt(result["sand_m3"])} m³
• Kerikil: sekitar {fmt(result["gravel_m3"])} m³
• Air: sekitar {fmt(result["water_liter"])} liter

🔍 VALIDASI

• Volume = panjang × lebar × tebal
• {fmt(p)} × {fmt(l)} × {fmt(t)} = {fmt(volume)} m³
• Estimasi material menggunakan asumsi 1 : 2 : 3

📊 RINGKASAN

• Beton: {fmt(volume)} m³
• Semen: {result["cement_sacks"]} zak
• Pasir: {fmt(result["sand_m3"])} m³
• Kerikil: {fmt(result["gravel_m3"])} m³
• Air: {fmt(result["water_liter"])} liter

📝 CATATAN

Hasil ini adalah estimasi kebutuhan material.
Untuk pekerjaan struktur penting, gunakan mix design
dan verifikasi engineer/insinyur struktur.
"""

    return answer.strip()


# ============================================================
# CIVIL WALL CALCULATOR
# ============================================================

def civil_wall_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 2:
        return None

    p, l = dims[0], dims[1]

    t = dims[2] if len(dims) >= 3 else None

    area = p * l

    lower = text.lower()

    if "batako" in lower:

        material_name = "batako"

        # Estimasi umum.
        # Nilai dapat berbeda sesuai ukuran batako.
        pieces_per_m2 = 12.5

    else:

        material_name = "bata"

        pieces_per_m2 = 50

    pieces = math.ceil(
        area * pieces_per_m2
    )

    # Mortar sederhana:
    mortar_per_m2 = 0.02

    mortar_volume = area * mortar_per_m2

    # Estimasi campuran mortar 1:4
    dry_factor = 1.33

    dry_mortar = (
        mortar_volume * dry_factor
    )

    cement_volume = dry_mortar / 5

    cement_kg = cement_volume * 1440

    cement_sacks = math.ceil(
        cement_kg / 50
    )

    sand = (
        dry_mortar * 4 / 5
    )

    return f"""
🏗️ KEBUTUHAN DINDING

📋 DATA

• Panjang dinding: {fmt(p)} m
• Tinggi dinding: {fmt(l)} m
• Luas dinding: {fmt(area)} m²
• Material: {material_name}

⚙️ ASUMSI

• Bata: sekitar 50 buah/m²
• Batako: sekitar 12,5 buah/m²
• Mortar: sekitar 0,02 m³/m²
• Campuran mortar: 1 : 4
• Faktor volume kering mortar: 1,33

🧮 PERHITUNGAN

• Luas = {fmt(p)} × {fmt(l)}
• Luas = {fmt(area)} m²
• {material_name}: sekitar {pieces} buah
• Mortar: sekitar {fmt(mortar_volume)} m³
• Semen: sekitar {fmt(cement_kg)} kg
• Semen: sekitar {cement_sacks} zak
• Pasir mortar: sekitar {fmt(sand)} m³

🔍 VALIDASI

• Luas dinding dihitung dari panjang × tinggi
• Jumlah material dibulatkan ke atas
• Angka adalah estimasi dan tergantung ukuran material nyata

📊 RINGKASAN

• Luas: {fmt(area)} m²
• {material_name}: sekitar {pieces} buah
• Semen: sekitar {cement_sacks} zak
• Pasir mortar: sekitar {fmt(sand)} m³

📝 CATATAN

Jumlah aktual dapat berubah karena ukuran bata/batako,
ketebalan nat, bukaan pintu/jendela, pecahan, dan metode
pemasangan.
""".strip()


# ============================================================
# PLASTER CALCULATOR
# ============================================================

def civil_plaster_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 2:
        return None

    p, h = dims[0], dims[1]

    area = p * h

    thickness = 0.015

    thickness_data = extract_value_with_unit(
        text,
        ["mm", "cm", "m"]
    )

    if thickness_data:

        value, unit = thickness_data

        candidate = to_meter(
            value,
            unit
        )

        if 0.003 <= candidate <= 0.1:
            thickness = candidate

    volume = area * thickness

    dry_factor = 1.33

    dry_volume = volume * dry_factor

    # 1:4
    cement_volume = dry_volume / 5

    sand_volume = (
        dry_volume * 4 / 5
    )

    cement_kg = (
        cement_volume * 1440
    )

    cement_sacks = math.ceil(
        cement_kg / 50
    )

    return f"""
🏗️ KEBUTUHAN PLESTERAN

📋 DATA

• Panjang: {fmt(p)} m
• Tinggi: {fmt(h)} m
• Luas: {fmt(area)} m²
• Tebal plester: {fmt(thickness * 1000)} mm

⚙️ ASUMSI

• Campuran mortar: 1 : 4
• Faktor volume kering: 1,33
• Berat semen: 1.440 kg/m³
• 1 zak semen: 50 kg

🧮 PERHITUNGAN

• Volume basah: {fmt(volume)} m³
• Volume kering: {fmt(dry_volume)} m³
• Semen: sekitar {fmt(cement_kg)} kg
• Semen: sekitar {cement_sacks} zak
• Pasir: sekitar {fmt(sand_volume)} m³

📊 RINGKASAN

• Luas: {fmt(area)} m²
• Semen: sekitar {cement_sacks} zak
• Pasir: sekitar {fmt(sand_volume)} m³
• Tebal: {fmt(thickness * 1000)} mm

📝 CATATAN

Hasil adalah estimasi. Konsumsi aktual dipengaruhi
ketebalan, permukaan dinding, campuran, dan kehilangan
material di lapangan.
""".strip()


# ============================================================
# ACIAN CALCULATOR
# ============================================================

def civil_acian_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 2:
        return None

    p, h = dims[0], dims[1]

    area = p * h

    # Estimasi konsumsi acian:
    # sekitar 1,5 kg/m2/mm
    thickness_mm = 2

    thickness_data = extract_value_with_unit(
        text,
        ["mm", "cm"]
    )

    if thickness_data:

        value, unit = thickness_data

        if unit == "cm":
            value *= 10

        if 1 <= value <= 10:
            thickness_mm = value

    consumption = (
        1.5 * thickness_mm
    )

    powder_kg = (
        area * consumption
    )

    bags = math.ceil(
        powder_kg / 40
    )

    return f"""
🏗️ KEBUTUHAN ACIAN

📋 DATA

• Panjang: {fmt(p)} m
• Tinggi: {fmt(h)} m
• Luas: {fmt(area)} m²
• Tebal acian: sekitar {fmt(thickness_mm)} mm

⚙️ ASUMSI

• Konsumsi acian: sekitar 1,5 kg/m²/mm
• Kemasan estimasi: 40 kg/zak

🧮 PERHITUNGAN

• Kebutuhan: {fmt(powder_kg)} kg
• Perkiraan: {bags} zak

📊 RINGKASAN

• Luas: {fmt(area)} m²
• Acian: sekitar {fmt(powder_kg)} kg
• Estimasi kemasan 40 kg: {bags} zak

📝 CATATAN

Konsumsi aktual mengikuti produk acian yang digunakan,
ketebalan aplikasi, dan kondisi permukaan.
""".strip()


# ============================================================
# EXCAVATION / URUGAN
# ============================================================

def civil_volume_calculation(
    text,
    title,
    material_name
):

    dims = extract_dimensions(text)

    if len(dims) < 3:
        return None

    p, l, t = dims[:3]

    volume = p * l * t

    return f"""
🏗️ {title}

📋 DATA

• Panjang: {fmt(p)} m
• Lebar: {fmt(l)} m
• Kedalaman/Tebal: {fmt(t)} m

🧮 PERHITUNGAN

Volume =
panjang × lebar × kedalaman

= {fmt(p)} × {fmt(l)} × {fmt(t)}

= {fmt(volume)} m³

📊 RINGKASAN

• {material_name}: {fmt(volume)} m³

📝 CATATAN

Angka merupakan volume geometris.
Kebutuhan pembelian aktual dapat berbeda karena
pemadatan, swell, penyusutan, dan kehilangan material.
""".strip()


# ============================================================
# REBAR CALCULATOR
# ============================================================

def civil_rebar_calculation(text):

    t = text.lower()

    diameter_match = re.search(
        r"(?:d|dia|diameter|besi)\s*"
        r"(\d+(?:[.,]\d+)?)\s*mm",
        t
    )

    if not diameter_match:

        diameter_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*mm",
            t
        )

    if not diameter_match:
        return None

    diameter = parse_number(
        diameter_match.group(1)
    )

    if not diameter:
        return None

    # Jumlah batang
    batang_match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:batang|btg)",
        t
    )

    jumlah_batang = None

    if batang_match:
        jumlah_batang = int(
            parse_number(
                batang_match.group(1)
            )
        )

    # Panjang total
    panjang_match = re.search(
        r"(?:panjang|total)\s*"
        r"(\d+(?:[.,]\d+)?)\s*m",
        t
    )

    panjang_total = None

    if panjang_match:
        panjang_total = parse_number(
            panjang_match.group(1)
        )

    # Jika hanya "besi D10 20 m"
    # dan tidak ditemukan keyword panjang
    if panjang_total is None:

        simple_length = re.search(
            r"(\d+(?:[.,]\d+)?)\s*m",
            t
        )

        if simple_length:
            panjang_total = parse_number(
                simple_length.group(1)
            )

    standard_length = 12

    standard_match = re.search(
        r"(?:batang|panjang batang|standar)"
        r"\s*(?:=|:)?\s*"
        r"(\d+(?:[.,]\d+)?)\s*m",
        t
    )

    if standard_match:

        standard_length = parse_number(
            standard_match.group(1)
        )

    weight_per_meter = (
        diameter ** 2 / 162
    )

    if jumlah_batang is not None:

        panjang_total = (
            jumlah_batang * standard_length
        )

    if panjang_total is None:
        return None

    weight_total = (
        panjang_total *
        weight_per_meter
    )

    if jumlah_batang is None:

        jumlah_batang = math.ceil(
            panjang_total /
            standard_length
        )

    bought_length = (
        jumlah_batang *
        standard_length
    )

    return f"""
🔩 KEBUTUHAN BESI

📋 DATA

• Diameter: D{fmt(diameter)} mm
• Panjang total: {fmt(panjang_total)} m
• Panjang batang standar: {fmt(standard_length)} m
• Jumlah batang: {jumlah_batang}

🧮 PERHITUNGAN

Rumus berat besi:

D² / 162

= {fmt(diameter)}² / 162

= {fmt(weight_per_meter)} kg/m

Berat total:

{fmt(panjang_total)} × {fmt(weight_per_meter)}
= {fmt(weight_total)} kg

📊 RINGKASAN

• Besi: D{fmt(diameter)} mm
• Panjang total: {fmt(panjang_total)} m
• Jumlah batang: {jumlah_batang} batang
• Total panjang pembelian: {fmt(bought_length)} m
• Berat estimasi: {fmt(weight_total)} kg

🔍 VALIDASI

✅ Rumus berat menggunakan D² / 162
✅ Panjang pembelian berdasarkan batang standar

📝 CATATAN

Perhitungan ini menghitung berat dan kebutuhan material besi.
Ini bukan desain tulangan struktur.

Jumlah dan diameter tulangan struktur harus ditentukan
berdasarkan perhitungan engineer.
""".strip()


# ============================================================
# FOOTPLAT
# ============================================================

def civil_footplat_calculation(text):

    dims = extract_dimensions(text)

    if len(dims) < 3:
        return None

    p, l, t = dims[:3]

    volume_one = p * l * t

    qty_match = re.search(
        r"(\d+)\s*(?:buah|bh|unit|titik|buah footplat)",
        text.lower()
    )

    quantity = 1

    if qty_match:
        quantity = int(
            qty_match.group(1)
        )

    total_volume = (
        volume_one * quantity
    )

    materials = concrete_materials(
        total_volume
    )

    return f"""
🏗️ KEBUTUHAN FOOTPLAT

📋 DATA

• Panjang: {fmt(p)} m
• Lebar: {fmt(l)} m
• Tebal: {fmt(t)} m
• Jumlah: {quantity} buah

🧮 PERHITUNGAN

Volume 1 footplat:

{fmt(p)} × {fmt(l)} × {fmt(t)}
= {fmt(volume_one)} m³

Total beton:

{fmt(volume_one)} × {quantity}
= {fmt(total_volume)} m³

🔩 ESTIMASI MATERIAL BETON

• Semen: sekitar {materials["cement_sacks"]} zak
• Pasir: sekitar {fmt(materials["sand_m3"])} m³
• Kerikil: sekitar {fmt(materials["gravel_m3"])} m³
• Air: sekitar {fmt(materials["water_liter"])} liter

🔍 VALIDASI

• Volume sudah dikalikan jumlah footplat.
• Material menggunakan asumsi beton 1 : 2 : 3.

⚠️ CATATAN STRUKTURAL

Ukuran footplat dan tulangannya tidak boleh dianggap aman
hanya dari volume beton.

Diperlukan data seperti:
• beban
• kondisi tanah
• jumlah lantai
• ukuran kolom
• mutu beton
• mutu baja
• kondisi pondasi

Untuk desain final, verifikasi engineer struktur diperlukan.
""".strip()


# ============================================================
# SLOOF / KOLOM / BALOK / PLAT
# ============================================================

def civil_structural_member_calculation(
    text,
    member_name
):

    dims = extract_dimensions(text)

    if len(dims) < 3:
        return None

    a, b, length = dims[:3]

    quantity_match = re.search(
        r"(\d+)\s*(?:buah|bh|unit)",
        text.lower()
    )

    quantity = 1

    if quantity_match:
        quantity = int(
            quantity_match.group(1)
        )

    volume_one = (
        a * b * length
    )

    total_volume = (
        volume_one * quantity
    )

    materials = concrete_materials(
        total_volume
    )

    return f"""
🏗️ KEBUTUHAN {member_name.upper()}

📋 DATA

• Dimensi penampang: {fmt(a)} m × {fmt(b)} m
• Panjang: {fmt(length)} m
• Jumlah: {quantity}
• Volume 1 elemen: {fmt(volume_one)} m³
• Total volume: {fmt(total_volume)} m³

⚙️ ASUMSI BETON

• Campuran nominal: 1 : 2 : 3
• Faktor kering: 1,54
• Semen: 1.440 kg/m³
• 1 zak: 50 kg
• w/c sekitar 0,50

🧮 ESTIMASI MATERIAL

• Semen: sekitar {fmt(materials["cement_kg"])} kg
• Semen: sekitar {materials["cement_sacks"]} zak
• Pasir: sekitar {fmt(materials["sand_m3"])} m³
• Kerikil: sekitar {fmt(materials["gravel_m3"])} m³
• Air: sekitar {fmt(materials["water_liter"])} liter

🔍 VALIDASI

• Volume = luas penampang × panjang × jumlah
• Setiap angka menggunakan satuan meter.

⚠️ CATATAN STRUKTURAL

Perhitungan di atas adalah kebutuhan volume/material awal.

Belum menentukan apakah ukuran dan tulangan
{member_name.lower()} aman.

Untuk desain struktur diperlukan data beban,
bentang, mutu beton, mutu baja, kondisi tumpuan,
dan ketentuan desain yang berlaku.

🎯 KESIMPULAN

Volume beton:
{fmt(total_volume)} m³

Semen:
sekitar {materials["cement_sacks"]} zak
""".strip()


# ============================================================
# CIVIL MASTER ROUTER
# ============================================================

def civil_calculator(text):

    t = text.lower().strip()

    # --------------------------------------------------------
    # BESI
    # --------------------------------------------------------

    if (
        "besi" in t
        or re.search(r"\bd\d+\b", t)
        or "diameter" in t
    ):
        result = civil_rebar_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # FOOTPLAT
    # --------------------------------------------------------

    if (
        "footplat" in t
        or "foot plate" in t
        or "tapak" in t
    ):
        result = civil_footplat_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # SLOOF
    # --------------------------------------------------------

    if "sloof" in t:

        result = civil_structural_member_calculation(
            text,
            "sloof"
        )

        if result:
            return result

    # --------------------------------------------------------
    # KOLOM
    # --------------------------------------------------------

    if "kolom" in t:

        result = civil_structural_member_calculation(
            text,
            "kolom"
        )

        if result:
            return result

    # --------------------------------------------------------
    # BALOK
    # --------------------------------------------------------

    if "balok" in t:

        result = civil_structural_member_calculation(
            text,
            "balok"
        )

        if result:
            return result

    # --------------------------------------------------------
    # PLAT
    # --------------------------------------------------------

    if (
        "plat beton" in t
        or "plat lantai" in t
    ):

        result = civil_concrete_calculation(
            text,
            "KEBUTUHAN PLAT BETON"
        )

        if result:
            return result

    # --------------------------------------------------------
    # PONDASI BATU KALI
    # --------------------------------------------------------

    if (
        "pondasi batu kali" in t
        or "batu kali" in t
    ):

        dims = extract_dimensions(text)

        if len(dims) >= 3:

            p, l, h = dims[:3]

            volume = p * l * h

            stone = volume * 1.20
            mortar = volume * 0.20

            cement_kg = (
                mortar * 1.33 / 5 * 1440
            )

            cement_sacks = math.ceil(
                cement_kg / 50
            )

            sand = (
                mortar * 1.33 * 4 / 5
            )

            return f"""
🏗️ KEBUTUHAN PONDASI BATU KALI

📋 DATA

• Panjang: {fmt(p)} m
• Lebar rata-rata: {fmt(l)} m
• Tinggi: {fmt(h)} m
• Volume geometris: {fmt(volume)} m³

⚙️ ASUMSI

• Batu kali: sekitar 1,20 × volume geometris
• Mortar: sekitar 20% volume geometris
• Mortar 1 : 4
• Faktor kering mortar: 1,33
• Semen: 1.440 kg/m³
• 1 zak: 50 kg

🧮 ESTIMASI MATERIAL

• Batu kali: sekitar {fmt(stone)} m³
• Mortar: sekitar {fmt(mortar)} m³
• Semen: sekitar {fmt(cement_kg)} kg
• Semen: sekitar {cement_sacks} zak
• Pasir mortar: sekitar {fmt(sand)} m³

🔍 VALIDASI

• Volume geometris = panjang × lebar × tinggi
• Batu dan mortar menggunakan faktor estimasi.

⚠️ CATATAN

Ukuran pondasi tidak boleh ditentukan hanya berdasarkan
volume material.

Kondisi tanah, beban bangunan, kedalaman pondasi,
dan dimensi aktual harus diperiksa untuk desain final.
""".strip()

    # --------------------------------------------------------
    # DINDING
    # --------------------------------------------------------

    if (
        "dinding" in t
        or "tembok" in t
        or "bata" in t
        or "batako" in t
    ):

        result = civil_wall_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # PLESTER
    # --------------------------------------------------------

    if "plester" in t:

        result = civil_plaster_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # ACIAN
    # --------------------------------------------------------

    if "acian" in t:

        result = civil_acian_calculation(text)

        if result:
            return result

    # --------------------------------------------------------
    # GALIAN
    # --------------------------------------------------------

    if (
        "galian" in t
        or "menggali" in t
    ):

        result = civil_volume_calculation(
            text,
            "VOLUME GALIAN",
            "volume galian"
        )

        if result:
            return result

    # --------------------------------------------------------
    # URUGAN
    # --------------------------------------------------------

    if (
        "urugan" in t
        or "urug" in t
    ):

        result = civil_volume_calculation(
            text,
            "VOLUME URUGAN",
            "volume urugan"
        )

        if result:
            return result

    # --------------------------------------------------------
    # BETON / LANTAI
    # --------------------------------------------------------

    concrete_keywords = [
        "beton",
        "lantai beton",
        "cor",
        "ngecor",
        "coran",
        "sipil",
    ]

    if any(
        x in t
        for x in concrete_keywords
    ):

        result = civil_concrete_calculation(
            text
        )

        if result:
            return result

    return None


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
        "sipil",
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

    math_keywords = [
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

    if any(x in t for x in reasoning):
        return "reasoning"

    if any(x in t for x in math_keywords):
        return "math"

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
            task
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
    ):
        model = GROQ_REASONING_MODEL

    else:
        model = GROQ_FAST_MODEL

    r = groq.chat.completions.create(
        model=model,
        messages=build_messages(
            uid,
            text,
            task
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

    # Civil Calculator deterministic
    if task == "civil":

        civil_result = civil_calculator(
            text
        )

        if civil_result:

            return (
                civil_result,
                "Civil Calculator",
                "Local Calculation Engine",
                task,
            )

    if task == "technical":

        providers = [
            (
                "OpenRouter Free",
                lambda: call_openrouter(
                    uid,
                    text,
                    task
                ),
            ),
            (
                "Gemini",
                lambda: call_gemini(
                    uid,
                    text,
                    task
                ),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(
                    uid,
                    text,
                    task
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
                    task
                ),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(
                    uid,
                    text,
                    task
                ),
            ),
            (
                "Gemini",
                lambda: call_gemini(
                    uid,
                    text,
                    task
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
                    task
                ),
            ),
            (
                "Gemini",
                lambda: call_gemini(
                    uid,
                    text,
                    task
                ),
            ),
            (
                "Groq Free Tier",
                lambda: call_groq(
                    uid,
                    text,
                    task
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

            if not answer.strip():
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
                f"{provider_name}: {error_text[:300]}"
            )

            log.warning(
                "PROVIDER FAILED | provider=%s | error=%s",
                provider_name,
                error_text[:300],
            )

    raise RuntimeError(
        "Semua provider AI GRATIS gagal. "
        + " | ".join(errors)
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
# TELEGRAM FORMATTER
# ============================================================

def clean_telegram_text(text):

    if not text:
        return "Tidak ada jawaban."

    text = str(text).replace(
        "\r\n",
        "\n"
    )

    text = re.sub(
        r"```[a-zA-Z0-9_+\-]*\n?",
        "",
        text,
    )

    text = text.replace(
        "```",
        ""
    )

    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = text.replace(
        "**",
        ""
    )

    text = re.sub(
        r"(?<!\w)\*(?!\s)",
        "",
        text,
    )

    text = text.replace(
        "__",
        ""
    )

    text = text.replace(
        "`",
        ""
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
                for cell in
                stripped.strip("|").split("|")
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
        "MATERIAL": "🔩",
        "SAMBUNGAN": "🔧",
        "WASTE": "♻️",
        "TRUE WASTE": "🗑️",
        "REUSABLE OFFCUT": "♻️",
        "CIVIL CALCULATOR": "🏗️",
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
# TELEGRAM CHUNK
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
    text
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
    data
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
    data
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
    prompt
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
                "OpenRouter Vision: "
                + str(e)[:220]
            )

    raise RuntimeError(
        "Semua provider vision gagal: "
        + " | ".join(errors)
    )


# ============================================================
# GEMINI VIDEO
# ============================================================

def analyze_video(
    data,
    mime,
    prompt
):

    if not gemini:
        raise RuntimeError(
            "Gemini diperlukan untuk video."
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
                None
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

def cloudflare_image(
    prompt
):

    if not CLOUDFLARE_ACCOUNT_ID:

        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID belum tersedia."
        )

    if not CLOUDFLARE_API_TOKEN:

        raise RuntimeError(
            "CLOUDFLARE_API_TOKEN belum tersedia."
        )

    url = (
        "https://api.cloudflare.com/client/v4/"
        f"accounts/{CLOUDFLARE_ACCOUNT_ID}/"
        f"ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    )

    with httpx.Client(
        timeout=300
    ) as client:

        r = client.post(
            url,
            headers={
                "Authorization":
                    f"Bearer {CLOUDFLARE_API_TOKEN}",
                "Content-Type":
                    "application/json",
            },
            json={
                "prompt": prompt
            },
        )

        if r.status_code >= 400:

            raise RuntimeError(
                f"Cloudflare HTTP "
                f"{r.status_code}: "
                f"{r.text[:400]}"
            )

        payload = r.json()

        if not payload.get("success"):

            raise RuntimeError(
                "Cloudflare gagal: "
                f"{payload.get('errors')}"
            )

        image_b64 = (
            payload
            .get("result", {})
            .get("image")
        )

        if not image_b64:

            raise RuntimeError(
                "Cloudflare mengembalikan data gambar kosong."
            )

        return base64.b64decode(
            image_b64
        )


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


def generate_image(
    prompt
):

    if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:

        try:

            return (
                cloudflare_image(
                    prompt
                ),
                "Cloudflare FLUX",
            )

        except Exception:

            log.exception(
                "Cloudflare gagal, fallback ke generator lama"
            )

    if POLLINATIONS_ENABLED:

        return (
            pollinations_image(
                prompt
            ),
            "Pollinations",
        )

    raise RuntimeError(
        "Generate gambar GRATIS belum tersedia."
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
            """
🤖 Designmanufaktur Super AI Agent aktif.

🧠 Smart Multi-AI Router
🏗️ Civil Calculator
✂️ Cutting List
🔩 Perhitungan Besi
🧱 Dinding
🏗️ Beton
🏠 Pondasi
🖼️ Gemini Vision
🎥 Gemini Video
🎨 Free Image Generation

🏗️ CIVIL CALCULATOR

✅ Beton
✅ Sloof
✅ Kolom
✅ Balok
✅ Plat beton
✅ Footplat
✅ Pondasi
✅ Batu kali
✅ Dinding bata
✅ Batako
✅ Plester
✅ Acian
✅ Galian
✅ Urugan
✅ Besi
✅ Berat besi
✅ Jumlah batang besi

✂️ CUTTING LIST

✅ Validasi jumlah potongan
✅ Validasi kapasitas batang
✅ Validasi total material
✅ True Waste
✅ Reusable Offcut
✅ Anti double-counting

Contoh:

sipil beton 5 x 10 meter tebal 10 cm

sloof 20 x 30 cm panjang 50 meter

kolom 30 x 30 cm tinggi 4 meter

plat beton 5 x 10 meter tebal 12 cm

footplat 1 x 1 x 0.3 meter 4 buah

dinding bata 10 x 3 meter

plester 10 x 3 meter tebal 15 mm

acian 10 x 3 meter

besi D10 panjang 120 meter

galian 10 x 1 x 0.8 meter

urugan 10 x 5 x 0.2 meter

Perintah:

/model
/reset
/gambar <prompt>

⚠️ Untuk struktur:
hasil material bukan pengganti desain engineer.
""",
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

        memory[uid] = []
        await save_persistent_memory(uid)

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

💰 PAID MODEL ROUTING
DISABLED
""",
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
                """
🎨 GENERATE GAMBAR

Contoh:

/gambar pagar minimalis hitam modern
""",
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
                "❌ Generate gambar gagal.\n"
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

            data, path = await tg_file(
                message["video"]["file_id"]
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
                    caption or
                    """
Analisa video ini secara detail.

Jika terkait pekerjaan manufaktur,
bengkel, konstruksi, sipil, tenda,
kanopi, atau fabrikasi:

- jelaskan objek
- jelaskan proses
- jelaskan kondisi
- jelaskan masalah
- berikan saran praktis

Jangan mengarang ukuran yang tidak terlihat.
""",
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
sipil, atau produk custom:

- jelaskan objek
- jelaskan komponen
- jelaskan fungsi
- jelaskan kondisi
- jelaskan masalah
- berikan saran praktis

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

        return

    # ========================================================
    # NORMAL CHAT
    # ========================================================

    if not text:
        return

    try:

        await load_persistent_memory(uid)

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

        await save_persistent_memory(uid)

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
            "Designmanufaktur Super AI Agent + Civil Calculator",

        "free_only":
            True,

        "telegram_format":
            "clean_and_mobile_friendly",

        "civil_calculator":
            True,

        "providers": {

            "gemini":
                bool(gemini),

            "openrouter_free":
                bool(openrouter),

            "groq_free_tier":
                bool(groq),

        },

        "civil_features": [

            "concrete",
            "sloof",
            "kolom",
            "balok",
            "plat",
            "footplat",
            "pondasi",
            "dinding",
            "plester",
            "acian",
            "galian",
            "urugan",
            "rebar",

        ],

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