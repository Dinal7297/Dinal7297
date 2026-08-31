# DESIGNMANUFAKTUR SUPER AI AGENT — FINAL ARCHITECTURE v7
# Existing application preserved; policy layer added.

import asyncio
import base64
import mimetypes
import os
import time
import random
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
OPENROUTER_FREE_MODEL = os.getenv(
    "OPENROUTER_FREE_MODEL",
    "openrouter/free",
)

GROQ_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_FAST_MODEL = os.getenv(
    "GROQ_FAST_MODEL",
    "openai/gpt-oss-20b"
)

GROQ_REASONING_MODEL = os.getenv(
    "GROQ_REASONING_MODEL",
    "openai/gpt-oss-120b"
)

GROQ_CODING_MODEL = os.getenv(
    "GROQ_CODING_MODEL",
    "qwen/qwen3-32b"
)

# ------------------------------------------------------------
# NVIDIA NIM / DeepSeek-V4-Flash-0731 (FREE TRIAL ENDPOINT)
# ------------------------------------------------------------
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "deepseek-ai/deepseek-v4-flash-0731",
)
NVIDIA_MAX_OUTPUT_TOKENS = int(os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "4096"))

# ------------------------------------------------------------
# MODEL CADANGAN TAMBAHAN (BARU)
# ------------------------------------------------------------
# Tidak dibaca dari env yang mungkin salah/sudah usang di hosting.
# Ini hanya lapisan cadangan EKSTRA supaya chat tetap jalan walau
# GROQ_FAST_MODEL / GROQ_REASONING_MODEL / GROQ_CODING_MODEL /
# OPENROUTER_FREE_MODEL yang di-set lewat env sudah tidak valid
# (model_not_found / dihapus provider).
GROQ_BACKUP_MODEL_1 = "openai/gpt-oss-20b"
GROQ_BACKUP_MODEL_2 = "openai/gpt-oss-120b"
GROQ_BACKUP_MODEL_3 = "qwen/qwen3-32b"
GROQ_BACKUP_MODEL_4 = "llama-3.1-8b-instant"
GROQ_BACKUP_MODEL_5 = "gemma2-9b-it"

OPENROUTER_BACKUP_MODEL = (
    "meta-llama/llama-3.3-70b-instruct:free"
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

# ------------------------------------------------------------
# CLOUDFLARE WORKERS AI (FLUX) - GENERATOR GAMBAR UTAMA (BARU)
# ------------------------------------------------------------

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    "",
)

CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    "",
)

CLOUDFLARE_IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell",
)

CLOUDFLARE_ENABLED = bool(
    CLOUDFLARE_ACCOUNT_ID
    and CLOUDFLARE_API_TOKEN
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

nvidia = (
    OpenAI(
        api_key=NVIDIA_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        timeout=30.0,
        max_retries=0,
    )
    if NVIDIA_KEY
    else None
)


# ============================================================
# MEMORY
# ============================================================

memory = {}
MAX_MEMORY = 20

# Mode AI per pengguna: auto / nvidia / groq / gemini / openrouter
user_ai_mode = {}

# ============================================================
# PERSISTENT MEMORY — SEPARATE GITHUB REPOSITORY
# ============================================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Dinal7297/designmanufaktur-memory")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_MEMORY_DIR = "memory"
GITHUB_API = "https://api.github.com"

# ============================================================
# UPLOAD HASIL PEKERJAAN — REPO WEBSITE (terpisah dari repo memory)
# ============================================================
# Reuse GITHUB_TOKEN yang sama (asal token itu juga punya akses ke
# repo website). Kalau perlu token berbeda, set WEBSITE_GITHUB_TOKEN
# di environment variable hosting.
WEBSITE_GITHUB_TOKEN = os.getenv("WEBSITE_GITHUB_TOKEN", GITHUB_TOKEN)
WEBSITE_GITHUB_REPO = os.getenv("WEBSITE_GITHUB_REPO", "Dinal7297/Design-Manufaktur-")
WEBSITE_GITHUB_BRANCH = os.getenv("WEBSITE_GITHUB_BRANCH", "main")
WEBSITE_DATA_PATH = "data/pekerjaan.json"

# ------------------------------------------------------------
# BATAS KONTEKS YANG DIKIRIM KE LLM (BARU)
# ------------------------------------------------------------
# Terpisah dari MAX_MEMORY (yang tetap dipakai untuk penyimpanan
# permanent memory di GitHub, TIDAK berubah). Ini hanya mengatur
# berapa banyak riwayat yang benar-benar dikirim ke provider AI
# per request, supaya tidak kena limit token/rate dari provider
# gratis (khususnya Groq yang TPM-nya kecil).
MAX_CONTEXT_TURNS = 8
MAX_CONTEXT_CHARS_PER_ITEM = 1200

GROQ_MAX_CONTEXT_TURNS = 4
GROQ_MAX_CONTEXT_CHARS_PER_ITEM = 400
GROQ_MAX_OUTPUT_TOKENS = 1536

OPENROUTER_MAX_OUTPUT_TOKENS = 2048


# FINAL v7 POLICY
FREE_ONLY_MODE=True
FAST_MEMORY_ENABLED=False
AGENT_MEMORY_ENABLED=True
AGENT_MEMORY_SOURCE="github"
MODEL_SELECTION_ENABLED=True
EXPERT_ROUTING_ENABLED=True
EXPERT_POLICY={"CHAT":["NVIDIA","GROQ","OPENROUTER_FREE","GEMINI"],"CODING":["NVIDIA_CODING","NVIDIA","GROQ","OPENROUTER_FREE","GEMINI"],"REASONING":["NVIDIA_REASONING","NVIDIA","GEMINI","GROQ","OPENROUTER_FREE"],"TECHNICAL":["NVIDIA","GEMINI","GROQ","OPENROUTER_FREE"],"CIVIL":["LOCAL_CIVIL","NVIDIA","GEMINI","GROQ","OPENROUTER_FREE"],"IMAGE":["GEMINI"],"VIDEO":["GEMINI"]}
def v7_memory_allowed(mode): return str(mode).upper()=="AGENT"
def v7_provider_allowed(provider): return (not FREE_ONLY_MODE) or str(provider).upper() in {"NVIDIA","NVIDIA_CODING","NVIDIA_REASONING","GROQ","OPENROUTER_FREE","GEMINI","LOCAL_CIVIL","LOCAL_CUTTING","POLLINATIONS"}


def history(uid):
    return memory.setdefault(uid, [])


def remember(uid, role, content):

    history(uid).append({
        "role": role,
        "content": content,
    })

    memory[uid] = history(uid)[-MAX_MEMORY:]


def _trim_history_for_context(
    uid,
    max_turns=MAX_CONTEXT_TURNS,
    max_chars_per_item=MAX_CONTEXT_CHARS_PER_ITEM,
):
    """
    Ambil riwayat terbaru saja (max_turns item terakhir),
    dan potong setiap item yang terlalu panjang.
    Tujuan: hemat token supaya tidak kena limit provider gratis.
    Tidak mengubah data yang tersimpan di memory[] / GitHub,
    hanya mempengaruhi apa yang dikirim ke LLM saat itu.
    """

    items = history(uid)[-max_turns:]

    trimmed = []

    for m in items:

        content = m.get("content", "") or ""

        if len(content) > max_chars_per_item:

            content = (
                content[:max_chars_per_item]
                + "\n...(riwayat dipotong untuk hemat token)"
            )

        trimmed.append({
            "role": m.get("role", "user"),
            "content": content,
        })

    return trimmed


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


# ============================================================
# UPLOAD HASIL PEKERJAAN KE REPO WEBSITE (fitur baru, aditif)
# ============================================================
# Alur: user kirim FOTO ke bot dengan caption diawali "/pekerjaan",
# format: /pekerjaan <kategori> <lokasi>
# contoh:  /pekerjaan kanopi cibinong
#
# Bot akan:
#  1) upload foto ke assets/pekerjaan/<kategori>/images/<slug>.jpg
#  2) tambahkan 1 entri baru ke data/pekerjaan.json
#  3) push ke branch main -> GitHub Action otomatis generate halaman
#
# Tidak menyentuh/mengubah data/pekerjaan.json manapun secara manual,
# hanya menambah 1 entri baru di akhir array lewat GitHub Contents API.

CATEGORY_FOLDER_MAP = {
    "kanopi": "kanopi",
    "pagar": "pagar",
    "pagar besi": "pagar",
    "pintu": "pintu",
    "teralis": "teralis",
    "tralis": "teralis",
    "railing": "railing",
    "tenda": "tenda",
}


def _slugify(text):
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "item"


async def _github_get_json(repo, path, token, branch):
    """Ambil isi file JSON dari GitHub. Return (data_list, sha) atau ([], None) kalau belum ada."""
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, headers=headers, params={"ref": branch})
    if resp.status_code == 404:
        return [], None
    resp.raise_for_status()
    body = resp.json()
    encoded = body.get("content", "")
    raw = base64.b64decode(encoded.replace("\n", "")).decode("utf-8") if encoded else "[]"
    data = json.loads(raw) if raw.strip() else []
    if not isinstance(data, list):
        data = []
    return data, body.get("sha")


async def _github_put_file(repo, path, content_bytes, message, token, branch, sha=None):
    """Simpan/replace 1 file (bisa teks atau gambar) ke GitHub lewat Contents API."""
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    encoded = base64.b64encode(content_bytes).decode("ascii")
    body = {"message": message, "content": encoded, "branch": branch}
    if sha:
        body["sha"] = sha
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.put(url, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()


def _build_pekerjaan_entry(category_raw, location_raw, image_path):
    category_slug = CATEGORY_FOLDER_MAP.get(
        category_raw.strip().lower(), _slugify(category_raw)
    )
    category_label = category_raw.strip().title()
    location_label = location_raw.strip().title()
    unique_suffix = str(int(time.time()))[-6:]
    slug = f"{_slugify(category_raw)}-{_slugify(location_raw)}-{unique_suffix}"

    today = time.strftime("%d %B %Y")
    today_iso = time.strftime("%Y-%m-%d")

    title = f"Pembuatan {category_label} di {location_label}"
    description = (
        f"Hasil pekerjaan pembuatan dan pemasangan {category_label.lower()} "
        f"oleh DESIGN MANUFAKTUR di {location_label}."
    )
    content = (
        f"<p>DESIGN MANUFAKTUR mengerjakan pembuatan {category_label.lower()} "
        f"sesuai kebutuhan pelanggan. Pekerjaan meliputi proses fabrikasi, "
        f"pengelasan, perakitan, finishing, dan pemasangan.</p>"
        f"<p>Proyek ini dikerjakan untuk kebutuhan bangunan di wilayah "
        f"{location_label}. Setiap pekerjaan dibuat berdasarkan ukuran dan "
        f"kebutuhan di lokasi.</p>"
    )

    return {
        "slug": slug,
        "title": title,
        "description": description,
        "date": today,
        "dateISO": today_iso,
        "category": category_label,
        "image": image_path,
        "imageAlt": f"{title} hasil pekerjaan DESIGN MANUFAKTUR",
        "url": f"/pekerjaan/{slug}/",
        "content": content,
    }, category_slug


async def upload_pekerjaan_from_photo(photo_bytes, category_raw, location_raw):
    """
    Upload 1 foto + 1 entri data pekerjaan baru ke repo website.
    Return dict berisi info hasil untuk ditampilkan ke user.
    """
    if not WEBSITE_GITHUB_TOKEN or not WEBSITE_GITHUB_REPO:
        raise RuntimeError(
            "WEBSITE_GITHUB_TOKEN / WEBSITE_GITHUB_REPO belum diset di environment variable."
        )

    # tentukan nama file & path dulu (butuh category_slug)
    category_slug_preview = CATEGORY_FOLDER_MAP.get(
        category_raw.strip().lower(), _slugify(category_raw)
    )
    filename_preview = f"{_slugify(category_raw)}-{_slugify(location_raw)}-{str(int(time.time()))[-6:]}.jpg"
    image_path = f"/assets/pekerjaan/{category_slug_preview}/images/{filename_preview}"

    entry, category_slug = _build_pekerjaan_entry(category_raw, location_raw, image_path)
    # pastikan filename di entry & file yang diupload konsisten
    filename = image_path.split("/")[-1]
    github_image_path = f"assets/pekerjaan/{category_slug}/images/{filename}"

    # 1) upload foto
    await _github_put_file(
        WEBSITE_GITHUB_REPO,
        github_image_path,
        photo_bytes,
        f"add: foto {entry['slug']}",
        WEBSITE_GITHUB_TOKEN,
        WEBSITE_GITHUB_BRANCH,
    )

    # 2) update data/pekerjaan.json (ambil dulu, tambahkan, simpan lagi)
    data_list, sha = await _github_get_json(
        WEBSITE_GITHUB_REPO, WEBSITE_DATA_PATH, WEBSITE_GITHUB_TOKEN, WEBSITE_GITHUB_BRANCH
    )
    data_list.append(entry)
    new_raw = json.dumps(data_list, ensure_ascii=False, indent=2).encode("utf-8")
    await _github_put_file(
        WEBSITE_GITHUB_REPO,
        WEBSITE_DATA_PATH,
        new_raw,
        f"add: data pekerjaan {entry['slug']}",
        WEBSITE_GITHUB_TOKEN,
        WEBSITE_GITHUB_BRANCH,
        sha=sha,
    )

    return entry


def build_messages(
    uid,
    text,
    task,
    max_turns=MAX_CONTEXT_TURNS,
    max_chars_per_item=MAX_CONTEXT_CHARS_PER_ITEM,
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
    ] + _trim_history_for_context(
        uid,
        max_turns,
        max_chars_per_item,
    ) + [
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
        "besi",
        "besi tulangan",
        "tulangan",
        "begel",
        "bendrat",
        "berat besi",
        "diameter besi",
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

def call_openrouter(uid, text, task, model=None):

    if not openrouter:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum tersedia."
        )

    selected = model or OPENROUTER_FREE_MODEL

    r = openrouter.chat.completions.create(
        model=selected,
        messages=build_messages(
            uid,
            text,
            task
        ),
        max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
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
            f"OpenRouter ({selected}) mengembalikan jawaban kosong."
        )

    selected_model = (
        getattr(r, "model", None)
        or selected
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

    for m in _trim_history_for_context(uid):

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
# NVIDIA NIM / DEEPSEEK V4 FLASH
# ============================================================

def call_nvidia(uid, text, task, reasoning_effort=None, model=None):

    if not nvidia:
        raise RuntimeError("NVIDIA_API_KEY belum tersedia.")

    effort = reasoning_effort
    if effort is None:
        effort = {
            "reasoning": "max",
            "math": "high",
            "coding": "high",
            "technical": "high",
            "civil": "high",
            "creative": "none",
            "general": "none",
        }.get(task, "none")

    # Gunakan format NVIDIA yang sama dengan contoh API key/user yang sudah
    # terbukti dipakai: reasoning_effort dikirim melalui chat_template_kwargs.
    # Ini sengaja tidak mengubah provider lain.
    r = nvidia.chat.completions.create(
        model=(model or NVIDIA_MODEL),
        messages=build_messages(
            uid,
            text,
            task,
            max_turns=MAX_CONTEXT_TURNS,
            max_chars_per_item=MAX_CONTEXT_CHARS_PER_ITEM,
        ),
        temperature=1,
        top_p=0.95,
        max_tokens=NVIDIA_MAX_OUTPUT_TOKENS,
        extra_body={
            "chat_template_kwargs": {
                "thinking": effort != "none",
                "reasoning_effort": effort,
            }
        },
        stream=False,
    )

    message = r.choices[0].message
    answer = getattr(message, "content", None) or ""

    if not answer.strip():
        raise RuntimeError(
            f"NVIDIA ({model or NVIDIA_MODEL}) mengembalikan jawaban kosong."
        )

    return answer, (model or NVIDIA_MODEL)


# ============================================================
# GROQ
# ============================================================

def call_groq(uid, text, task, model=None):

    if not groq:
        raise RuntimeError(
            "GROQ_API_KEY belum tersedia."
        )

    if not model:

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
            max_turns=GROQ_MAX_CONTEXT_TURNS,
            max_chars_per_item=GROQ_MAX_CONTEXT_CHARS_PER_ITEM,
        ),
        max_tokens=GROQ_MAX_OUTPUT_TOKENS,
    )

    answer = (
        r.choices[0].message.content
        or ""
    )

    if not answer.strip():
        raise RuntimeError(
            f"Groq ({model}) mengembalikan jawaban kosong."
        )

    return answer, model


def _is_retryable_rate_limit(error_text):
    """
    True hanya untuk rate limit yang kemungkinan bersifat sesaat
    (per-menit/per-request), BUKAN limit harian/kuota yang sudah
    pasti habis (retry tidak akan membantu untuk itu, cuma buang waktu).
    """

    t = error_text.lower()

    if "per-day" in t or "per day" in t or "daily" in t:
        return False

    if "429" in t or "rate limit" in t or "resource_exhausted" in t:
        return True

    return False


def _call_with_retry(fn, retries=1, base_delay=1.5):

    last_err = None

    for attempt in range(retries + 1):

        try:

            return fn()

        except Exception as e:

            last_err = e

            if (
                attempt == retries
                or not _is_retryable_rate_limit(str(e))
            ):
                raise

            time.sleep(
                base_delay + random.uniform(0, 1.0)
            )

    raise last_err


# ============================================================
# SMART CHAT ROUTER
# ============================================================

def _provider_available(name):
    return {
        "nvidia": bool(nvidia),
        "groq": bool(groq),
        "gemini": bool(gemini),
        "openrouter": bool(openrouter),
    }.get(name, False)


def _provider_call(name, uid, text, task, model=None, reasoning_effort=None):
    if name == "nvidia":
        return call_nvidia(uid, text, task, reasoning_effort=reasoning_effort, model=model)
    if name == "groq":
        return call_groq(uid, text, task, model=model)
    if name == "gemini":
        return call_gemini(uid, text, task)
    if name == "openrouter":
        return call_openrouter(uid, text, task, model=model or OPENROUTER_FREE_MODEL)
    raise RuntimeError(f"Provider tidak dikenal: {name}")



def _provider_label(name):
    return {
        "nvidia": "NVIDIA DeepSeek V4 Flash",
        "groq": "Groq",
        "gemini": "Gemini",
        "openrouter": "OpenRouter Free",
    }.get(name, name)


def _auto_provider_order(task):
    if task in ("reasoning", "math", "coding", "technical", "civil"):
        return ["nvidia", "groq", "openrouter", "gemini"]
    return ["nvidia", "groq", "openrouter", "gemini"]


MANUAL_AI_MODES = {
    "nvidia_fast": ("nvidia", NVIDIA_MODEL, "none", "🟢 NVIDIA Fast"),
    "nvidia_coding": ("nvidia", NVIDIA_MODEL, "high", "💻 NVIDIA Coding"),
    "nvidia_technical": ("nvidia", NVIDIA_MODEL, "high", "🔧 NVIDIA Technical"),
    "nvidia_reasoning": ("nvidia", NVIDIA_MODEL, "max", "🧠 NVIDIA Reasoning"),
    "groq_fast": ("groq", GROQ_FAST_MODEL, None, "⚡ Groq Fast"),
    "groq_coding": ("groq", GROQ_CODING_MODEL, None, "💻 Groq Coding"),
    "groq_reasoning": ("groq", GROQ_REASONING_MODEL, None, "🧠 Groq Reasoning"),
    "gemini": ("gemini", GEMINI_CHAT_MODEL, None, "👁 Gemini"),
    "openrouter": ("openrouter", OPENROUTER_FREE_MODEL, None, "🌐 OpenRouter FREE"),
}

def _transparent_error_reason(exc):
    """Ringkas error provider tanpa membocorkan API key atau payload."""
    t = str(exc).lower()
    if "401" in t or "unauthorized" in t or "invalid api key" in t or "authentication" in t:
        return "AUTH"
    if "403" in t or "forbidden" in t:
        return "FORBIDDEN"
    if "429" in t or "rate limit" in t or "resource_exhausted" in t or "quota" in t:
        return "RATE LIMIT/QUOTA"
    if "timeout" in t or "timed out" in t:
        return "TIMEOUT"
    if "model_not_found" in t or "model not found" in t or "does not exist" in t:
        return "MODEL TIDAK TERSEDIA"
    if "422" in t or "validation" in t:
        return "VALIDATION"
    return "ERROR"


def _transparent_route_line(provider_name, model, status, reason=None):
    icons = {"nvidia": "🟢", "groq": "⚡", "gemini": "👁️", "openrouter": "🌐"}
    icon = icons.get(provider_name, "🤖")
    label = _provider_label(provider_name)
    suffix = f" — {reason}" if reason else ""
    return f"{icon} {label} | {model or 'default'} | {status}{suffix}"


def chat_router(uid, text):
    started = time.perf_counter()
    task = classify_task(text)
    mode = user_ai_mode.get(str(uid), "auto")

    if task == "civil":
        civil_result = civil_calculator(text)
        if civil_result:
            elapsed = time.perf_counter() - started
            meta = {
                "provider": "local",
                "provider_label": "Civil Calculator",
                "model": "Local Calculation Engine",
                "task": task,
                "status": "LOCAL",
                "elapsed": elapsed,
                "routes": ["🧮 Civil Calculator | Local Calculation Engine | LOCAL"],
            }
            return civil_result, meta

    if mode == "auto":
        routes = [(p, None, None) for p in _auto_provider_order(task)]
    else:
        cfg = MANUAL_AI_MODES.get(mode)
        if not cfg:
            mode = "auto"
            routes = [(p, None, None) for p in _auto_provider_order(task)]
        else:
            provider, model, effort, _label = cfg
            fallback = ["nvidia", "groq", "openrouter", "gemini"]
            routes = [(provider, model, effort)] + [(p, None, None) for p in fallback if p != provider]

    attempts = []
    for provider_name, selected_model, effort in routes:
        if not _provider_available(provider_name):
            attempts.append({
                "provider": provider_name,
                "model": selected_model,
                "status": "SKIP",
                "reason": "API TIDAK AKTIF",
            })
            continue
        try:
            answer, model = _call_with_retry(
                lambda p=provider_name, m=selected_model, e=effort: _provider_call(
                    p, uid, text, task, model=m, reasoning_effort=e
                ),
                retries=1,
            )
            if not answer.strip():
                raise RuntimeError("Provider mengembalikan jawaban kosong.")
            status = "MAIN" if not attempts else "FALLBACK"
            attempts.append({
                "provider": provider_name,
                "model": model,
                "status": "SUCCESS",
            })
            elapsed = time.perf_counter() - started
            meta = {
                "provider": provider_name,
                "provider_label": _provider_label(provider_name),
                "model": model,
                "task": task,
                "status": status,
                "elapsed": elapsed,
                "routes": attempts,
            }
            return answer, meta
        except Exception as e:
            attempts.append({
                "provider": provider_name,
                "model": selected_model,
                "status": "FAILED",
                "reason": _transparent_error_reason(e),
            })

    elapsed = time.perf_counter() - started
    detail = "\n".join(
        f"• {_provider_label(a['provider'])}: {a.get('reason', a.get('status', 'UNKNOWN'))}"
        for a in attempts
    )
    raise RuntimeError("Semua provider AI GRATIS gagal.\n" + detail)


def format_transparent_response(answer, meta):
    """Header wajib yang menjelaskan AI/model/task/status/waktu/riwayat."""
    if meta.get("provider") == "local":
        history = "\n".join(f"• {x}" for x in meta.get("routes", []))
    else:
        history_lines = []
        for a in meta.get("routes", []):
            if a.get("status") == "FAILED":
                history_lines.append(
                    _transparent_route_line(
                        a["provider"], a.get("model"), "❌ GAGAL", a.get("reason")
                    )
                )
            elif a.get("status") == "SKIP":
                history_lines.append(
                    _transparent_route_line(
                        a["provider"], a.get("model"), "⏭️ DILEWATI", a.get("reason")
                    )
                )
            elif a.get("status") == "SUCCESS":
                history_lines.append(
                    _transparent_route_line(
                        a["provider"], a.get("model"), "✅ BERHASIL"
                    )
                )
        history = "\n".join(history_lines) if history_lines else "• Tidak ada percobaan provider."

    status_icon = "🎯" if meta.get("status") == "MAIN" else ("🔄" if meta.get("status") == "FALLBACK" else "🧮")
    provider_display = meta.get("provider_label", "Unknown")
    model_display = meta.get("model", "Unknown")
    task_display = str(meta.get("task", "general")).upper()
    elapsed = meta.get("elapsed", 0.0)

    header = (
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 AI: {provider_display}\n"
        f"🧠 Model: {model_display}\n"
        f"📌 Task: {task_display}\n"
        f"{status_icon} Status: {meta.get('status', 'UNKNOWN')}\n"
        f"⏱️ Waktu: {elapsed:.2f} detik\n"
        "🔍 Riwayat routing:\n"
        f"{history}\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
    )
    return header + answer

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
    data,
    filename="image.png",
    content_type="image/png",
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
                    filename,
                    data,
                    content_type,
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

    started = time.perf_counter()
    if not gemini and not openrouter:
        raise RuntimeError("Tidak ada provider vision GRATIS yang aktif.")

    attempts = []

    if gemini:
        try:
            r = gemini.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=[
                    types.Part.from_bytes(data=data, mime_type=mime),
                    SYSTEM + "\n\n" + prompt,
                ],
            )
            if r.text:
                attempts.append({"provider": "gemini", "model": GEMINI_CHAT_MODEL, "status": "SUCCESS"})
                return r.text, GEMINI_CHAT_MODEL, {
                    "provider": "gemini",
                    "provider_label": "Gemini",
                    "model": GEMINI_CHAT_MODEL,
                    "task": "vision",
                    "status": "MAIN" if not attempts[:-1] else "FALLBACK",
                    "elapsed": time.perf_counter() - started,
                    "routes": attempts,
                }
            raise RuntimeError("jawaban kosong")
        except Exception as e:
            attempts.append({"provider": "gemini", "model": GEMINI_CHAT_MODEL, "status": "FAILED", "reason": _transparent_error_reason(e)})

    if openrouter:
        try:
            b64 = base64.b64encode(data).decode()
            content = [
                {"type": "text", "text": SYSTEM + "\n\n" + prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]
            r = openrouter.chat.completions.create(
                model=OPENROUTER_FREE_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
            )
            answer = r.choices[0].message.content or ""
            if answer.strip():
                selected = getattr(r, "model", None) or OPENROUTER_FREE_MODEL
                attempts.append({"provider": "openrouter", "model": selected, "status": "SUCCESS"})
                return answer, selected, {
                    "provider": "openrouter",
                    "provider_label": "OpenRouter Free",
                    "model": selected,
                    "task": "vision",
                    "status": "FALLBACK" if attempts[:-1] else "MAIN",
                    "elapsed": time.perf_counter() - started,
                    "routes": attempts,
                }
            raise RuntimeError("jawaban kosong")
        except Exception as e:
            attempts.append({"provider": "openrouter", "model": OPENROUTER_FREE_MODEL, "status": "FAILED", "reason": _transparent_error_reason(e)})

    detail = "\n".join(
        f"• {_provider_label(a['provider'])}: {a.get('reason', a.get('status', 'UNKNOWN'))}"
        for a in attempts
    )
    raise RuntimeError("Semua provider vision GRATIS gagal.\n" + detail)


# ============================================================
# GEMINI VIDEO
# ============================================================

def analyze_video(
    data,
    mime,
    prompt
):

    started = time.perf_counter()
    if not gemini:
        raise RuntimeError("Gemini diperlukan untuk video.")

    uploaded = gemini.files.upload(
        file=types.Part.from_bytes(data=data, mime_type=mime)
    )

    for _ in range(60):
        f = gemini.files.get(name=uploaded.name)
        state = getattr(getattr(f, "state", None), "name", "")
        if state == "ACTIVE":
            uploaded = f
            break
        if state == "FAILED":
            raise RuntimeError("Gemini gagal memproses video.")
        time.sleep(2)
    else:
        raise RuntimeError("Video belum siap diproses.")

    result = gemini.models.generate_content(
        model=GEMINI_CHAT_MODEL,
        contents=[uploaded, SYSTEM + "\n\n" + prompt],
    )
    answer = result.text or ""
    if not answer.strip():
        raise RuntimeError("Gemini mengembalikan jawaban video kosong.")

    meta = {
        "provider": "gemini",
        "provider_label": "Gemini",
        "model": GEMINI_CHAT_MODEL,
        "task": "video",
        "status": "MAIN",
        "elapsed": time.perf_counter() - started,
        "routes": [{"provider": "gemini", "model": GEMINI_CHAT_MODEL, "status": "SUCCESS"}],
    }
    return answer, meta


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


def _to_jpeg_bytes(raw: bytes) -> bytes:
    """
    Konversi bytes gambar apapun (PNG/WEBP/dll) menjadi JPEG.
    Jika Pillow tidak tersedia atau gagal decode,
    kembalikan data asli apa adanya (tidak menggagalkan proses).
    """

    try:

        from PIL import Image
        import io as _io

        img = Image.open(
            _io.BytesIO(raw)
        )

        if img.mode in (
            "RGBA",
            "P",
            "LA",
        ):
            img = img.convert("RGB")

        out = _io.BytesIO()

        img.save(
            out,
            format="JPEG",
            quality=92,
        )

        return out.getvalue()

    except Exception:

        return raw


def cloudflare_flux_image(
    prompt
):
    """
    Generator gambar utama (BARU): Cloudflare Workers AI - FLUX.
    Tidak menggantikan generator lama, hanya ditambahkan di depan.
    """

    if not CLOUDFLARE_ENABLED:

        raise RuntimeError(
            "Cloudflare Workers AI belum dikonfigurasi "
            "(CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN kosong)."
        )

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
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
                f"Cloudflare Workers AI HTTP "
                f"{r.status_code}: "
                f"{r.text[:400]}"
            )

        content_type = r.headers.get(
            "content-type",
            "",
        )

        if content_type.startswith(
            "image/"
        ):

            raw = r.content

        else:

            payload = r.json()

            if not payload.get(
                "success",
                True,
            ):

                raise RuntimeError(
                    "Cloudflare Workers AI gagal: "
                    f"{payload.get('errors')}"
                )

            result = (
                payload.get("result")
                or {}
            )

            b64 = result.get("image")

            if not b64:

                raise RuntimeError(
                    "Cloudflare Workers AI mengembalikan "
                    "data tidak dikenal."
                )

            raw = base64.b64decode(b64)

        if not raw:

            raise RuntimeError(
                "Cloudflare Workers AI mengembalikan data kosong."
            )

        return _to_jpeg_bytes(raw)


def generate_image(
    prompt
):
    """
    Urutan generator gambar:
    1. Cloudflare Workers AI (FLUX) - utama (BARU)
    2. Pollinations - fallback (generator lama, TIDAK dihapus)
    """

    errors = []

    if CLOUDFLARE_ENABLED:

        try:

            return (
                cloudflare_flux_image(
                    prompt
                ),
                "Cloudflare Workers AI (FLUX)",
            )

        except Exception as e:

            log.exception(
                "cloudflare flux image generation failed"
            )

            errors.append(
                f"Cloudflare: {e}"
            )

    if POLLINATIONS_ENABLED:

        try:

            return (
                pollinations_image(
                    prompt
                ),
                "Pollinations",
            )

        except Exception as e:

            log.exception(
                "pollinations image generation failed"
            )

            errors.append(
                f"Pollinations: {e}"
            )

    raise RuntimeError(
        "Generate gambar GRATIS belum tersedia.\n"
        + "\n".join(errors)
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

    # ========================================================
    # INLINE MODEL PICKER CALLBACK
    # ========================================================
    callback = update.get("callback_query")
    if callback:
        callback_id = callback.get("id")
        data = callback.get("data", "")
        from_user = callback.get("from", {})
        uid = str(from_user.get("id", ""))
        chat_id = callback.get("message", {}).get("chat", {}).get("id")

        mode_map = {
            "ai_auto": "auto",
            "ai_nvidia_fast": "nvidia_fast",
            "ai_nvidia_coding": "nvidia_coding",
            "ai_nvidia_technical": "nvidia_technical",
            "ai_nvidia_reasoning": "nvidia_reasoning",
            "ai_groq_fast": "groq_fast",
            "ai_groq_coding": "groq_coding",
            "ai_groq_reasoning": "groq_reasoning",
            "ai_gemini": "gemini",
            "ai_openrouter": "openrouter",
        }
        if data in mode_map:
            user_ai_mode[uid] = mode_map[data]
            if callback_id:
                try:
                    await tg("answerCallbackQuery", {
                        "callback_query_id": callback_id,
                        "text": f"Mode AI: {mode_map[data].upper()}",
                    })
                except Exception:
                    pass
            if chat_id:
                label = ("🔄 AUTO" if mode_map[data] == "auto" else MANUAL_AI_MODES.get(mode_map[data], (None, None, None, mode_map[data]))[3])
                await send_text(
                    chat_id,
                    f"✅ Mode AI diubah ke: {label}\n\n"
                    "Mode ini menjadi AI utama. Jika gagal/limit, "
                    "bot tetap otomatis pindah ke provider GRATIS lain.",
                )
        return

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

📸 UPLOAD HASIL PEKERJAAN KE WEBSITE
Kirim foto dengan caption:
/pekerjaan <kategori> <lokasi>
Contoh: /pekerjaan kanopi cibinong

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

        current_mode = user_ai_mode.get(str(uid), "auto")
        keyboard = {
            "inline_keyboard": [
                [{"text": "🔄 AUTO", "callback_data": "ai_auto"}],
                [{"text": "🟢 NVIDIA Fast", "callback_data": "ai_nvidia_fast"}, {"text": "💻 NVIDIA Coding", "callback_data": "ai_nvidia_coding"}],
                [{"text": "🔧 NVIDIA Technical", "callback_data": "ai_nvidia_technical"}, {"text": "🧠 NVIDIA Reasoning", "callback_data": "ai_nvidia_reasoning"}],
                [{"text": "⚡ Groq Fast", "callback_data": "ai_groq_fast"}, {"text": "💻 Groq Coding", "callback_data": "ai_groq_coding"}],
                [{"text": "🧠 Groq Reasoning", "callback_data": "ai_groq_reasoning"}],
                [{"text": "👁 Gemini", "callback_data": "ai_gemini"}, {"text": "🌐 OpenRouter FREE", "callback_data": "ai_openrouter"}],
            ]
        }

        current_label = "🔄 AUTO" if current_mode == "auto" else MANUAL_AI_MODES.get(current_mode, (None,None,None,current_mode))[3]
        status = (
            f"🤖 MODE AI: {current_label}\n\n"
            f"NVIDIA: {'✅ AKTIF' if nvidia else '❌ TIDAK AKTIF'}\n"
            f"Groq: {'✅ AKTIF' if groq else '❌ TIDAK AKTIF'}\n"
            f"Gemini: {'✅ AKTIF' if gemini else '❌ TIDAK AKTIF'}\n"
            f"OpenRouter FREE: {'✅ AKTIF' if openrouter else '❌ TIDAK AKTIF'}\n\n"
            "Pilih AI/model utama. Fallback GRATIS tetap aktif.\n\n"
            f"NVIDIA: {NVIDIA_MODEL}\n"
            f"Gemini: {GEMINI_CHAT_MODEL}\n"
            f"Groq Coding: {GROQ_CODING_MODEL}\n"
            f"Groq Reasoning: {GROQ_REASONING_MODEL}\n"
            f"Groq Fast: {GROQ_FAST_MODEL}\n"
            f"OpenRouter: {OPENROUTER_FREE_MODEL}\n\n"
            "💰 PAID MODEL ROUTING: DISABLED"
        )

        await tg(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": clean_telegram_text(status),
                "reply_markup": keyboard,
            },
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

            if provider.startswith(
                "Cloudflare"
            ):

                await send_photo(
                    chat_id,
                    data,
                    filename="image.jpg",
                    content_type="image/jpeg",
                )

            else:

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

            answer, video_meta = await asyncio.to_thread(
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

            await send_text(
                chat_id,
                format_transparent_response(answer, video_meta),
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
    # UPLOAD HASIL PEKERJAAN (foto + caption "/pekerjaan <kategori> <lokasi>")
    # ========================================================

    if message.get("photo") and caption.strip().lower().startswith("/pekerjaan"):

        args = command_arg(caption.strip())
        parts = args.split(maxsplit=1)

        if len(parts) < 2:
            await send_text(
                chat_id,
                "Format caption belum lengkap.\n\n"
                "Kirim foto dengan caption:\n"
                "/pekerjaan <kategori> <lokasi>\n\n"
                "Contoh:\n/pekerjaan kanopi cibinong",
            )
            return

        category_raw, location_raw = parts[0], parts[1]

        await send_text(
            chat_id,
            f"⏳ Mengupload foto & menambahkan hasil pekerjaan "
            f"({category_raw} - {location_raw})...",
        )

        try:
            photo_bytes, _ = await tg_file(message["photo"][-1]["file_id"])

            entry = await upload_pekerjaan_from_photo(
                photo_bytes, category_raw, location_raw
            )

            await send_text(
                chat_id,
                "✅ Berhasil ditambahkan!\n\n"
                f"Judul: {entry['title']}\n"
                f"Kategori: {entry['category']}\n\n"
                "Halaman akan otomatis dibuat dalam 1-2 menit "
                "(GitHub Action sedang generate halaman).\n"
                f"Link nanti: https://design-manufaktur.vercel.app{entry['url']}",
            )

        except Exception as e:
            log.exception("upload pekerjaan failed")
            await send_text(
                chat_id,
                "❌ Gagal upload hasil pekerjaan.\n" + str(e)[:700],
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

            answer, model, vision_meta = await asyncio.to_thread(
                analyze_image,
                data,
                mime,
                prompt,
            )

            await send_text(
                chat_id,
                format_transparent_response(answer, vision_meta),
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

        answer, meta = await asyncio.to_thread(
            chat_router,
            uid,
            text,
        )

        remember(
            uid,
            "user",
            text,
        )

        # Simpan jawaban mentah agar header transparansi tidak ikut masuk memory.
        remember(
            uid,
            "assistant",
            answer,
        )

        await save_persistent_memory(uid)

        transparent_answer = format_transparent_response(answer, meta)

        await send_text(
            chat_id,
            transparent_answer,
        )

        log.info(
            "CHAT DONE | task=%s | provider=%s | model=%s | status=%s | elapsed=%.2fs | routes=%s",
            meta.get("task"),
            meta.get("provider_label"),
            meta.get("model"),
            meta.get("status"),
            meta.get("elapsed", 0.0),
            meta.get("routes"),
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

            "nvidia_free_trial":
                bool(nvidia),

            "cloudflare_flux":
                CLOUDFLARE_ENABLED,

            "pollinations_fallback":
                POLLINATIONS_ENABLED,

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

            "nvidia":
                NVIDIA_MODEL,

            "cloudflare_image":
                CLOUDFLARE_IMAGE_MODEL,

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