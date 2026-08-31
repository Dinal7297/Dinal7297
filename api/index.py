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
from calculators.civil import civil_calculator
from calculators.cutting import CUTTING_TASK_HINT
from calculators.technical import TECHNICAL_TASK_HINT


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

NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "deepseek-ai/deepseek-v4-flash-0731"
)
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)
NVIDIA_MAX_OUTPUT_TOKENS = int(
    os.getenv("NVIDIA_MAX_OUTPUT_TOKENS", "2048")
)

# ------------------------------------------------------------
# MODEL CADANGAN TAMBAHAN (BARU)
# ------------------------------------------------------------
# Tidak dibaca dari env yang mungkin salah/sudah usang di hosting.
# Ini hanya lapisan cadangan EKSTRA supaya chat tetap jalan walau
# OPENROUTER_FREE_MODEL yang di-set lewat env sudah tidak valid
# (model_not_found / dihapus provider).

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
GAYA NATURAL & EKSPRESIF
============================================================

Buat jawaban terasa seperti asisten manusia yang ramah dan membantu.
Gunakan emoji secara natural dan secukupnya, terutama: 👍 😊 🔧 📐 🧮 🏗️ ✂️ ⚙️ ⚠️ ✅ 🎯.
Jangan memakai emoji di setiap kalimat dan jangan berlebihan.
Untuk jawaban teknis, emoji berfungsi sebagai penanda bagian agar mudah dibaca di HP.
Tetap profesional, jelas, dan jangan mengorbankan akurasi demi gaya.

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

nvidia = (
    OpenAI(
        api_key=NVIDIA_KEY,
        base_url=NVIDIA_BASE_URL,
    )
    if NVIDIA_KEY
    else None
)

# ------------------------------------------------------------
# GROK / XAI — fallback tambahan untuk FAST/AGENT
# ------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", ""))
GROK_MODEL = os.getenv("GROK_MODEL", "grok-4.1-fast")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
GROK_MAX_OUTPUT_TOKENS = int(os.getenv("GROK_MAX_OUTPUT_TOKENS", "1200"))

grok = (
    OpenAI(
        api_key=XAI_API_KEY,
        base_url=GROK_BASE_URL,
    )
    if XAI_API_KEY
    else None
)


# ============================================================
# MEMORY
# ============================================================

memory = {}
MAX_MEMORY = 20

# ============================================================
# PEMILIHAN AI MANUAL + TRANSPARANSI (BARU — ADITIF)
# ============================================================
# Tidak menghapus/mengubah sistem smart router yang sudah ada.
# Fitur ini hanya menambahkan:
#   1) Cara user memaksa pakai 1 provider/gaya tertentu
#   2) Info di setiap jawaban: AI mana yang berhasil / gagal,
#      lengkap dengan riwayat routing seperti /model.

AI_MODE_AUTO = "auto"

# gaya NVIDIA (task_hint yang dipaksa, model tetap sama)
NVIDIA_STYLE_TASK_MAP = {
    "nvidia_fast": "general",
    "nvidia_coding": "coding",
    "nvidia_technical": "technical",
    "nvidia_reasoning": "reasoning",
}

AI_MODE_CHOICES = {
    "auto": "auto",
    "otomatis": "auto",
    "nvidia": "nvidia_fast",
    "nvidia_fast": "nvidia_fast",
    "nvidia fast": "nvidia_fast",
    "nvidia_coding": "nvidia_coding",
    "nvidia coding": "nvidia_coding",
    "nvidia_technical": "nvidia_technical",
    "nvidia technical": "nvidia_technical",
    "nvidia_reasoning": "nvidia_reasoning",
    "nvidia reasoning": "nvidia_reasoning",
    "gemini": "gemini",
    "openrouter": "openrouter",
    "open router": "openrouter",
    "or": "openrouter",
}

AI_MODE_LABELS = {
    "auto": "🔁 Otomatis (Smart Router + fallback)",
    "nvidia_fast": "⚡ NVIDIA Fast",
    "nvidia_coding": "💻 NVIDIA Coding",
    "nvidia_technical": "🔧 NVIDIA Technical",
    "nvidia_reasoning": "🧠 NVIDIA Reasoning",
    "gemini": "👁️ Gemini",
    "openrouter": "🌐 OpenRouter FREE",
}

# uid(str) -> salah satu key di AI_MODE_LABELS
user_ai_mode = {}

# Mode chat: FAST tidak membaca memory cloud sebelum menjawab.
# AGENT membaca memory GitHub sebelum setiap chat dan menyimpan permanen.
CHAT_MODE_FAST = "fast"
CHAT_MODE_AGENT = "agent"
user_chat_mode = {}
user_tool_mode = {}
_memory_save_locks = {}
_memory_save_tasks = {}
_processed_update_ids = set()


def get_chat_mode(uid):
    return user_chat_mode.get(str(uid), CHAT_MODE_FAST)


def set_chat_mode(uid, mode):
    uid = str(uid)
    user_chat_mode[uid] = mode if mode in (CHAT_MODE_FAST, CHAT_MODE_AGENT) else CHAT_MODE_FAST
    # Beralih ke FAST/AGENT otomatis menutup mode kalkulator.
    user_tool_mode.pop(uid, None)


def get_tool_mode(uid):
    return user_tool_mode.get(str(uid))


def set_tool_mode(uid, mode):
    if mode not in ("civil", "cutting", "technical"):
        user_tool_mode.pop(str(uid), None)
    else:
        user_tool_mode[str(uid)] = mode


def get_ai_mode(uid):
    return user_ai_mode.get(str(uid), AI_MODE_AUTO)


def set_ai_mode(uid, mode):
    user_ai_mode[str(uid)] = mode


def build_ai_mode_keyboard():
    """Inline keyboard untuk memilih AI secara manual (tombol)."""

    def btn(mode):
        return {
            "text": AI_MODE_LABELS[mode],
            "callback_data": f"aimode:{mode}",
        }

    return {
        "inline_keyboard": [
            [btn("auto")],
            [btn("nvidia_fast"), btn("nvidia_coding")],
            [btn("nvidia_technical"), btn("nvidia_reasoning")],
            [btn("gemini"), btn("openrouter")],
        ]
    }


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
# gratis.
MAX_CONTEXT_TURNS = 8
MAX_CONTEXT_CHARS_PER_ITEM = 1200

NVIDIA_MAX_CONTEXT_TURNS = 6
NVIDIA_MAX_CONTEXT_CHARS_PER_ITEM = 800

OPENROUTER_MAX_OUTPUT_TOKENS = 2048


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
            saved_mode = saved.get("ai_mode")
            if saved_mode in AI_MODE_LABELS:
                user_ai_mode[uid] = saved_mode
            saved_chat_mode = saved.get("chat_mode")
            if saved_chat_mode in (CHAT_MODE_FAST, CHAT_MODE_AGENT):
                user_chat_mode[uid] = saved_chat_mode
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
    raw = json.dumps({"user_id": uid, "memory": history(uid)[-MAX_MEMORY:], "ai_mode": get_ai_mode(uid), "updated_at": int(time.time())}, ensure_ascii=False, indent=2)
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


async def background_save_memory(uid):
    uid = str(uid)
    lock = _memory_save_locks.setdefault(uid, asyncio.Lock())
    async with lock:
        await save_persistent_memory(uid)


def schedule_memory_save(uid):
    uid = str(uid)
    old_task = _memory_save_tasks.get(uid)
    if old_task and not old_task.done():
        return old_task
    task = asyncio.create_task(background_save_memory(uid))
    _memory_save_tasks[uid] = task
    return task


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
    include_history=True,
):

    task_hint = {
        "coding": "TUGAS CODING. Prioritaskan kode yang dapat dijalankan, ketepatan sintaks, debugging, struktur program, dan solusi praktis.",
        "reasoning": "TUGAS REASONING. Analisis masalah secara sistematis dan validasi kesimpulan.",
        "technical": TECHNICAL_TASK_HINT,
        "cutting": CUTTING_TASK_HINT,
        "civil": "TUGAS CIVIL CALCULATOR. Prioritaskan volume, dimensi, kebutuhan material, semen, pasir, kerikil, air, besi, berat besi, pondasi, dinding, plester, acian, galian, dan urugan. Jangan mengarang data.",
        "math": "TUGAS MATEMATIKA. Hitung teliti, tampilkan rumus penting, gunakan satuan, dan periksa hasil.",
        "creative": "TUGAS KREATIF. Buat hasil siap digunakan, praktis, menarik, dan sesuai tujuan.",
        "general": "TUGAS UMUM. Jawab langsung, jelas, dan berguna.",
    }.get(task, "")

    messages = [{"role": "system", "content": SYSTEM + "\n\n" + task_hint}]
    if include_history:
        messages.extend(_trim_history_for_context(uid, max_turns, max_chars_per_item))
    messages.append({"role": "user", "content": text})
    return messages

# ============================================================
# NUMBER PARSER
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

def call_openrouter(uid, text, task, model=None, include_history=True):

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
            task,
            include_history=include_history,
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

def call_gemini(uid, text, task, include_history=True):

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

    if include_history:
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
# NVIDIA
# ============================================================

def call_nvidia(uid, text, task, model=None, include_history=True):

    if not nvidia:
        raise RuntimeError(
            "NVIDIA_API_KEY belum tersedia."
        )

    selected = model or NVIDIA_MODEL

    r = nvidia.chat.completions.create(
        model=selected,
        messages=build_messages(
            uid,
            text,
            task,
            max_turns=NVIDIA_MAX_CONTEXT_TURNS,
            max_chars_per_item=NVIDIA_MAX_CONTEXT_CHARS_PER_ITEM,
            include_history=include_history,
        ),
        max_tokens=NVIDIA_MAX_OUTPUT_TOKENS,
        temperature=1,
        top_p=0.95,
    )

    answer = (
        r.choices[0].message.content
        or ""
    )

    if not answer.strip():
        raise RuntimeError(
            f"NVIDIA ({selected}) mengembalikan jawaban kosong."
        )

    selected_model = (
        getattr(r, "model", None)
        or selected
    )

    return answer, selected_model



# ============================================================
# GROK / XAI
# ============================================================

def call_grok(uid, text, task, include_history=True):
    if not grok:
        raise RuntimeError("XAI_API_KEY/GROK_API_KEY belum tersedia.")

    r = grok.chat.completions.create(
        model=GROK_MODEL,
        messages=build_messages(
            uid, text, task,
            max_turns=6,
            max_chars_per_item=800,
            include_history=include_history,
        ),
        max_tokens=GROK_MAX_OUTPUT_TOKENS,
        temperature=0.7,
    )
    answer = r.choices[0].message.content or ""
    if not answer.strip():
        raise RuntimeError(f"Grok ({GROK_MODEL}) mengembalikan jawaban kosong.")
    return answer, (getattr(r, "model", None) or GROK_MODEL)

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

def chat_router(uid, text, ai_mode="auto", include_history=True, forced_task=None):
    start_time = time.time()

    # FAST adalah jalur AI murni: tidak menjalankan classifier/tool otomatis.
    # Caller mengirim forced_task="general" dan include_history=False.
    task = forced_task if forced_task else classify_task(text)

    log.info("TASK=%s | ai_mode=%s | memory=%s | text=%s", task, ai_mode, include_history, text[:120])

    # Civil hanya dijalankan ketika shortcut /civil mengaktifkan forced_task,
    # atau ketika caller memang secara eksplisit meminta task civil.
    if forced_task == "civil":
        civil_result = civil_calculator(text)
        if civil_result:
            return (civil_result, "Civil Calculator", "Local Calculation Engine", task, [], round(time.time() - start_time, 2))

    # Prioritas otomatis: NVIDIA -> Gemini -> OpenRouter.
    # NVIDIA menjadi provider utama sesuai konfigurasi FAST/AGENT.
    # OpenRouter hanya dipakai jika dua provider utama gagal.
    providers = [
        ("⚡ NVIDIA", lambda: call_nvidia(uid, text, task, include_history=include_history)),
        ("👁️ Gemini", lambda: call_gemini(uid, text, task, include_history=include_history)),
        ("🚀 Grok", lambda: call_grok(uid, text, task, include_history=include_history)),
        ("🌐 OpenRouter Free", lambda: call_openrouter(uid, text, task, include_history=include_history)),
    ]

    if ai_mode in NVIDIA_STYLE_TASK_MAP:
        forced_nvidia_task = NVIDIA_STYLE_TASK_MAP[ai_mode]
        providers = [(AI_MODE_LABELS[ai_mode], lambda: call_nvidia(uid, text, forced_nvidia_task, include_history=include_history))]
    elif ai_mode == "gemini":
        providers = [("👁️ Gemini", lambda: call_gemini(uid, text, task, include_history=include_history))]
    elif ai_mode == "openrouter":
        providers = [("🌐 OpenRouter Free", lambda: call_openrouter(uid, text, task, include_history=include_history))]

    errors = []
    attempts = []
    for provider_name, fn in providers:
        try:
            # Jangan retry di provider utama: jika gagal, pindah langsung
            # ke provider berikutnya agar fallback tidak menambah delay.
            answer, model = fn()
            if not answer.strip():
                raise RuntimeError("Provider mengembalikan jawaban kosong.")
            attempts.append({"provider": provider_name, "model": model, "status": "ok"})
            return (answer, provider_name, model, task, attempts, round(time.time() - start_time, 2))
        except Exception as e:
            error_text = str(e)
            errors.append(f"{provider_name}: {error_text[:300]}")
            attempts.append({"provider": provider_name, "model": None, "status": "failed", "error": error_text[:200]})
            log.warning("PROVIDER FAILED | provider=%s | error=%s", provider_name, error_text[:300])

    raise RuntimeError("Semua provider AI GRATIS gagal. " + " | ".join(errors))

# ============================================================
# TELEGRAM API
# ============================================================

async def tg(method, data):

    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "TELEGRAM_TOKEN belum diatur."
        )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(12.0, connect=5.0)
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
    text,
    reply_markup=None,
):

    formatted = clean_telegram_text(
        text
    )

    chunks = split_telegram_message(
        formatted,
        max_length=3900,
    )

    for i, chunk in enumerate(chunks):

        payload = {
            "chat_id": chat_id,
            "text": chunk,
        }

        # reply_markup (mis. inline keyboard) hanya dilampirkan
        # di chunk TERAKHIR supaya tombol tidak terpotong-potong
        # kalau jawabannya panjang dan harus dikirim beberapa pesan.
        if reply_markup and i == len(chunks) - 1:
            payload["reply_markup"] = reply_markup

        await tg(
            "sendMessage",
            payload,
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

# ============================================================
# TRANSPARANSI JAWABAN (BARU)
# ============================================================
# Menampilkan AI mana yang benar-benar menjawab, model apa yang
# dipakai, AI mana saja yang sempat dicoba tapi gagal (lengkap
# dengan alasan singkatnya), berapa lama waktu prosesnya, dan
# mode AI yang sedang aktif (otomatis / manual).

def _short_reason(error_text):

    t = (error_text or "").lower()

    if "belum tersedia" in t:
        return "NO_KEY"

    if (
        "401" in t
        or "unauthorized" in t
        or "invalid api key" in t
        or "auth" in t
        or "403" in t
    ):
        return "AUTH"

    if (
        "429" in t
        or "rate limit" in t
        or "resource_exhausted" in t
        or "quota" in t
    ):
        return "RATE_LIMIT"

    if "timeout" in t or "timed out" in t:
        return "TIMEOUT"

    if "kosong" in t:
        return "EMPTY"

    return "ERROR"


def _build_transparency_footer(
    provider,
    model,
    task,
    ai_mode,
    attempts,
    elapsed,
):

    mode_label = AI_MODE_LABELS.get(
        ai_mode,
        AI_MODE_LABELS[AI_MODE_AUTO],
    )

    if provider == "Civil Calculator":

        return (
            "――――――――――――――――――――\n"
            "🧮 Dihitung oleh: Civil Calculator\n"
            "📌 Mesin hitung lokal (bukan AI), jadi tidak "
            "dipengaruhi mode AI.\n"
            f"⏱️ Waktu: {elapsed} detik"
        )

    attempts = attempts or []

    status_label = (
        "LANGSUNG BERHASIL"
        if len(attempts) <= 1
        else "FALLBACK"
    )

    lines = [
        "――――――――――――――――――――",
        f"🤖 AI: {provider}",
        f"🧠 Model: {model}",
        f"📌 Task: {task.upper()}",
        f"🔄 Status: {status_label}",
        f"⏱️ Waktu: {elapsed} detik",
        f"🎯 Mode AI: {mode_label}",
        "🔍 Riwayat routing:",
    ]

    for a in attempts:

        name = a.get("provider", "?")

        if a.get("status") == "ok":

            lines.append(
                f"{name} — {a.get('model')} — ✅ BERHASIL"
            )

        else:

            reason = _short_reason(
                a.get("error", "")
            )

            lines.append(
                f"{name} — default — ❌ GAGAL — {reason}"
            )

    lines.append("――――――――――――――――――――")

    return "\n".join(lines)


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
# HANDLE CALLBACK QUERY (tombol inline /model)
# ============================================================

async def handle_callback_query(callback_query):

    data = callback_query.get("data", "") or ""

    chat_id = (
        callback_query.get("message", {})
        .get("chat", {})
        .get("id")
    )

    uid = str(
        callback_query.get("from", {})
        .get("id", chat_id)
    )

    cq_id = callback_query.get("id")

    if not data.startswith("aimode:"):

        if cq_id:
            try:
                await tg(
                    "answerCallbackQuery",
                    {"callback_query_id": cq_id},
                )
            except Exception:
                pass

        return

    mode = data.split(":", 1)[1]

    if mode not in AI_MODE_LABELS:

        if cq_id:
            try:
                await tg(
                    "answerCallbackQuery",
                    {
                        "callback_query_id": cq_id,
                        "text": "❌ Pilihan tidak dikenali.",
                        "show_alert": False,
                    },
                )
            except Exception:
                pass

        return

    set_ai_mode(uid, mode)
    schedule_memory_save(uid)

    label = AI_MODE_LABELS[mode]

    if cq_id:
        try:
            await tg(
                "answerCallbackQuery",
                {
                    "callback_query_id": cq_id,
                    "text": f"Mode diubah ke {label}",
                },
            )
        except Exception:
            pass

    if chat_id:
        await send_text(
            chat_id,
            f"✅ Mode AI diubah ke:\n{label}",
        )


# ============================================================
# HANDLE TELEGRAM UPDATE
# ============================================================

async def handle(update):

    callback_query = update.get("callback_query")

    if callback_query:
        await handle_callback_query(callback_query)
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

SHORTCUT:

/fast — ⚡ AI cepat, AI-only, tanpa memory/tool
/agent — 🧠 AI Agent + memory GitHub permanen
/tools — 🧰 pilih/aktifkan tool
/model — pilih AI
/calc — menu kalkulator
/civil — Civil Calculator
/cutting — Cutting List
/technical — Technical Calculator
/gambar <prompt> — generate gambar
/reset — reset memory

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
        schedule_memory_save(uid)

        await send_text(
            chat_id,
            "✅ Memory sesi dihapus.",
        )

        return

    # ========================================================
    # CHAT MODE: FAST / AGENT
    # ========================================================

    if text.startswith("/fast"):
        set_chat_mode(uid, CHAT_MODE_FAST)
        set_tool_mode(uid, None)
        await send_text(chat_id, "⚡ FAST MODE AKTIF\n\nAI saja. Tidak membaca/menulis memory GitHub sebelum atau sesudah jawaban.\nTool tidak aktif otomatis.\nPrioritas: NVIDIA → Gemini → Grok → OpenRouter FREE.")
        return

    if text.startswith("/agent"):
        set_chat_mode(uid, CHAT_MODE_AGENT)
        set_tool_mode(uid, None)
        await load_persistent_memory(uid)
        await send_text(chat_id, "🧠 AGENT MODE AKTIF\n\nMemory lama dari GitHub dimuat dan percakapan Agent disimpan permanen.\nTool tetap tidak aktif otomatis.\nPrioritas: NVIDIA → Gemini → Grok → OpenRouter FREE.")
        return

    if text.startswith("/tools"):
        await send_text(chat_id, "🧰 MENU TOOLS\n\n/civil → 🏗️ Civil Calculator\n/cutting → ✂️ Cutting List\n/technical → 🔧 Technical Calculator\n/gambar → 🎨 Generate gambar\n/foto → 🖼️ Analisis foto\n/video → 🎥 Analisis video\n/pekerjaan → 📸 Upload hasil pekerjaan\n\nTool hanya aktif ketika command-nya dipanggil. Chat biasa tidak mengaktifkan tool otomatis.")
        return

    # ========================================================
    # MODEL
    # ========================================================

    if text.startswith(
        "/model"
    ):

        current_mode = get_ai_mode(uid)
        current_label = AI_MODE_LABELS.get(
            current_mode,
            AI_MODE_LABELS[AI_MODE_AUTO],
        )

        status_text = f"""
🤖 MODE AI: {current_label}
💬 CHAT MODE: {get_chat_mode(uid).upper()}

NVIDIA: {'✅ AKTIF' if nvidia else '❌ TIDAK AKTIF'}
Gemini: {'✅ AKTIF' if gemini else '❌ TIDAK AKTIF'}
OpenRouter FREE: {'✅ AKTIF' if openrouter else '❌ TIDAK AKTIF'}

NVIDIA: {NVIDIA_MODEL}
Gemini: {GEMINI_CHAT_MODEL}
OpenRouter: {OPENROUTER_FREE_MODEL}

💰 PAID MODEL ROUTING: DISABLED

🏗️ CIVIL CALCULATOR
✅ ACTIVE (Local Calculation Engine, bukan AI)

🎨 GENERATE GAMBAR
Cloudflare Flux: {'✅ AKTIF' if CLOUDFLARE_ENABLED else '❌ TIDAK AKTIF'}
Pollinations (fallback): {'✅ AKTIF' if POLLINATIONS_ENABLED else '❌ TIDAK AKTIF'}

👇 Tap tombol untuk ganti AI secara manual:
"""

        await send_text(
            chat_id,
            status_text,
            reply_markup=build_ai_mode_keyboard(),
        )

        return

    # ========================================================
    # AI (BARU) — pilih manual AI mana yang dipakai + transparansi
    # ========================================================

    if text.startswith(
        "/ai"
    ):

        arg = command_arg(text).strip().lower()

        if not arg:

            current_mode = get_ai_mode(uid)
            current_label = AI_MODE_LABELS.get(
                current_mode,
                AI_MODE_LABELS[AI_MODE_AUTO],
            )

            await send_text(
                chat_id,
                f"""
🤖 PEMILIHAN AI MANUAL

Mode saat ini:
{current_label}

Pilihan:

/ai auto
→ otomatis: NVIDIA → Gemini → Grok → OpenRouter FREE

/ai nvidia_fast
→ paksa pakai NVIDIA (gaya cepat/umum)

/ai nvidia_coding
→ paksa pakai NVIDIA (gaya coding)

/ai nvidia_technical
→ paksa pakai NVIDIA (gaya teknik/manufaktur)

/ai nvidia_reasoning
→ paksa pakai NVIDIA (gaya analisis mendalam)
{'✅ API key aktif' if nvidia else '❌ NVIDIA_API_KEY belum di-set'}

/ai openrouter
→ paksa pakai OpenRouter saja
{'✅ API key aktif' if openrouter else '❌ OPENROUTER_API_KEY belum di-set'}

/ai gemini
→ paksa pakai Gemini saja
{'✅ API key aktif' if gemini else '❌ GEMINI_API_KEY belum di-set'}

📌 CATATAN
• Civil Calculator (rumus beton, besi, dll) selalu pakai mesin hitung lokal, bukan AI, jadi tidak dipengaruhi mode ini.
• Mode manual TIDAK otomatis pindah ke AI lain kalau AI pilihanmu gagal, supaya kamu tahu persis AI mana yang gagal.
• Setiap jawaban chat akan menampilkan riwayat routing lengkap: AI mana yang menjawab, AI mana yang sempat gagal, dan berapa lama waktunya.
• Bisa juga pakai tombol lewat /model.
""",
                reply_markup=build_ai_mode_keyboard(),
            )

            return

        mode = AI_MODE_CHOICES.get(arg)

        if not mode:

            await send_text(
                chat_id,
                "❌ Pilihan tidak dikenali.\n\n"
                "Gunakan salah satu:\n"
                "/ai auto\n"
                "/ai nvidia_fast\n"
                "/ai nvidia_coding\n"
                "/ai nvidia_technical\n"
                "/ai nvidia_reasoning\n"
                "/ai openrouter\n"
                "/ai gemini\n\n"
                "Atau tap tombol lewat /model.",
            )

            return

        set_ai_mode(uid, mode)

        schedule_memory_save(uid)

        await send_text(
            chat_id,
            f"✅ Mode AI diubah ke:\n{AI_MODE_LABELS[mode]}",
        )

        return

    # ========================================================
    # TOOL SHORTCUTS — AKTIF HANYA SAAT DIPANGGIL
    # ========================================================

    if text.startswith("/calc"):
        await send_text(
            chat_id,
            "🧮 MENU KALKULATOR\n\n"
            "/civil → Civil Calculator\n"
            "/cutting → Cutting List\n"
            "/technical → Technical Calculator\n\n"
            "Kalkulator tidak aktif pada chat biasa. Pilih shortcut terlebih dahulu.",
        )
        return

    if text.startswith("/civil"):
        set_tool_mode(uid, "civil")
        await send_text(chat_id, "🏗️ CIVIL CALCULATOR AKTIF\n\nKirim data perhitungan civil berikutnya.\nMode akan tetap aktif sampai Anda memilih /fast atau /agent.")
        return

    if text.startswith("/cutting"):
        set_tool_mode(uid, "cutting")
        await send_text(chat_id, "✂️ CUTTING LIST AKTIF\n\nKirim daftar ukuran/potongan dan panjang batang.\nMode akan tetap aktif sampai Anda memilih /fast atau /agent.")
        return

    if text.startswith("/technical"):
        set_tool_mode(uid, "technical")
        await send_text(chat_id, "🔧 TECHNICAL CALCULATOR AKTIF\n\nKirim data teknis yang ingin dihitung/validasi.\nMode akan tetap aktif sampai Anda memilih /fast atau /agent.")
        return

    # ========================================================
    # TOOL SHORTCUTS — TIDAK AKTIF PADA CHAT BIASA
    # ========================================================

    if text.startswith("/calc"):
        await send_text(
            chat_id,
            "🧮 MENU KALKULATOR\n\n"
            "/civil → Civil Calculator\n"
            "/cutting → Cutting List\n"
            "/technical → Technical Calculator\n\n"
            "Pilih shortcut di atas. Kalkulator tidak aktif otomatis pada chat biasa.",
        )
        return

    if text.startswith("/civil"):
        set_tool_mode(uid, "civil")
        await send_text(chat_id, "🏗️ CIVIL CALCULATOR AKTIF\n\nKirim data perhitungan berikutnya.\nUntuk kembali ke chat AI, gunakan /fast atau /agent.")
        return

    if text.startswith("/cutting"):
        set_tool_mode(uid, "cutting")
        await send_text(chat_id, "✂️ CUTTING LIST AKTIF\n\nKirim daftar ukuran/potongan dan panjang batang.\nUntuk kembali ke chat AI, gunakan /fast atau /agent.")
        return

    if text.startswith("/technical"):
        set_tool_mode(uid, "technical")
        await send_text(chat_id, "🔧 TECHNICAL MODE AKTIF\n\nKirim data teknis yang ingin dihitung atau divalidasi.\nUntuk kembali ke chat AI, gunakan /fast atau /agent.")
        return

    if text.startswith("/status"):
        await send_text(
            chat_id,
            f"📊 STATUS BOT\n\n"
            f"Chat Mode: {get_chat_mode(uid).upper()}\n"
            f"AI Mode: {AI_MODE_LABELS.get(get_ai_mode(uid), get_ai_mode(uid))}\n"
            f"Tool Mode: {(get_tool_mode(uid) or 'none').upper()}\n\n"
            f"Gemini: {'✅ AKTIF' if gemini else '❌ TIDAK AKTIF'}\n"
            f"NVIDIA: {'✅ AKTIF' if nvidia else '❌ TIDAK AKTIF'}\n"
            f"OpenRouter: {'✅ AKTIF' if openrouter else '❌ TIDAK AKTIF'}",
        )
        return

    if text.startswith("/help"):
        await send_text(chat_id, "📖 SHORTCUT\n\n/fast — AI cepat tanpa load memory\n/agent — AI Agent dengan memory GitHub\n/model — pilih AI\n/calc — menu kalkulator\n/civil — Civil Calculator\n/cutting — Cutting List\n/technical — Technical Calculator\n/gambar <prompt> — generate gambar\n/pekerjaan <kategori> <lokasi> — upload hasil pekerjaan\n/reset — reset memory")
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

        chat_mode = get_chat_mode(uid)

        if chat_mode == CHAT_MODE_AGENT:
            await load_persistent_memory(uid)

        await tg(
            "sendChatAction",
            {"chat_id": chat_id, "action": "typing"},
        )

        ai_mode = get_ai_mode(uid)
        tool_mode = get_tool_mode(uid)

        # FAST benar-benar AI-only: jangan classifier, jangan tool, jangan memory.
        if chat_mode == CHAT_MODE_FAST:
            forced_task = "general"
            include_history = False
        else:
            forced_task = tool_mode if tool_mode in ("civil", "cutting", "technical") else None
            include_history = True

        (answer, provider, model, task, attempts, elapsed) = await asyncio.to_thread(
            chat_router, uid, text, ai_mode, include_history, forced_task
        )

        if chat_mode == CHAT_MODE_AGENT:
            remember(uid, "user", text)
            remember(uid, "assistant", answer)

        footer = _build_transparency_footer(
            provider, model, task, ai_mode, attempts, elapsed
        )

        # Satu pesan saja: jawaban + transparansi. Tidak ada double response.
        await send_text(chat_id, answer + "\n\n" + footer)

        # Hanya AGENT yang menyimpan permanen. FAST tidak menyentuh GitHub.
        if chat_mode == CHAT_MODE_AGENT:
            schedule_memory_save(uid)

        log.info(
            "CHAT DONE | task=%s | provider=%s | model=%s | mode=%s | waktu=%s",
            task,
            provider,
            model,
            ai_mode,
            elapsed,
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

            "nvidia_free_tier":
                bool(nvidia),

            "grok":
                bool(grok),

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

            "nvidia":
                NVIDIA_MODEL,

            "grok":
                GROK_MODEL,

            "cloudflare_image":
                CLOUDFLARE_IMAGE_MODEL,

        },
    }


TELEGRAM_COMMANDS = [
    {"command": "start", "description": "Mulai dan lihat bantuan"},
    {"command": "fast", "description": "⚡ Mode cepat tanpa memory"},
    {"command": "agent", "description": "🧠 Mode Agent dengan memory"},
    {"command": "tools", "description": "🧰 Pilih dan aktifkan tool"},
    {"command": "model", "description": "🤖 Pilih AI"},
    {"command": "calc", "description": "🧮 Menu kalkulator"},
    {"command": "civil", "description": "🏗️ Civil Calculator"},
    {"command": "cutting", "description": "✂️ Cutting List"},
    {"command": "technical", "description": "🔧 Technical Calculator"},
    {"command": "gambar", "description": "🎨 Generate gambar"},
    {"command": "foto", "description": "🖼️ Analisis foto"},
    {"command": "video", "description": "🎥 Analisis video"},
    {"command": "pekerjaan", "description": "📸 Upload hasil pekerjaan"},
    {"command": "reset", "description": "🧹 Reset memory"},
    {"command": "status", "description": "📊 Status bot"},
    {"command": "help", "description": "📖 Bantuan"},
]

@app.on_event("startup")
async def setup_telegram_commands():
    try:
        await tg("setMyCommands", {"commands": TELEGRAM_COMMANDS})
        log.info("TELEGRAM COMMAND MENU UPDATED | count=%s", len(TELEGRAM_COMMANDS))
    except Exception as e:
        log.warning("TELEGRAM COMMAND MENU UPDATE FAILED | %s", str(e)[:300])


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

    # Telegram dapat mengirim ulang update jika webhook terlalu lama.
    # Deduplicate berdasarkan update_id dan proses di background.
    update_id = update.get("update_id")
    if update_id is not None:
        if update_id in _processed_update_ids:
            return {"ok": True, "duplicate": True}
        _processed_update_ids.add(update_id)
        if len(_processed_update_ids) > 2000:
            # set sederhana: hapus sebagian ID tertua secara deterministik.
            for old_id in sorted(_processed_update_ids)[:500]:
                _processed_update_ids.discard(old_id)

    asyncio.create_task(handle(update))
    return {"ok": True}


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