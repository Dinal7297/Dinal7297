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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("designmanufaktur")
app = FastAPI(title="Designmanufaktur Super AI Agent")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Dinal7297/designmanufaktur-memory")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

GEMINI_CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
OPENROUTER_FREE_MODEL = os.getenv("OPENROUTER_FREE_MODEL", "openrouter/free")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
GROQ_REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b")
GROQ_CODING_MODEL = os.getenv("GROQ_CODING_MODEL", "qwen/qwen3-32b")
CLOUDFLARE_TEXT_MODEL = os.getenv("CLOUDFLARE_TEXT_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
openrouter = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1") if OPENROUTER_KEY else None
groq = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_KEY else None
cloudflare_text = OpenAI(api_key=CLOUDFLARE_API_TOKEN, base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1") if CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN else None
huggingface = OpenAI(api_key=HUGGINGFACE_API_KEY, base_url="https://router.huggingface.co/v1") if HUGGINGFACE_API_KEY else None

SYSTEM = """
Kamu adalah Designmanufaktur Super AI Agent untuk bengkel las, kanopi, pagar, tenda, rangka baja, hollow, pipa, fabrikasi, manufaktur, konstruksi ringan, pekerjaan sipil, beton, pondasi, sloof, kolom, balok, plat lantai, dinding, plesteran, acian, besi tulangan, cutting list, estimasi material, engineering, coding, bisnis, konten, pemasaran.

Jawab dalam Bahasa Indonesia kecuali diminta lain. Langsung ke inti, praktis, jelas, tidak bertele-tele, nyaman dibaca di HP, gunakan satuan yang jelas, hasil bisa dipakai untuk pekerjaan lapangan.

Jangan gunakan Markdown berlebihan. Hindari **bold**, *italic*, ###, ---, tabel dengan karakter |.

Gunakan heading dengan emoji: 📋 DATA, ⚙️ ASUMSI, 🧮 PERHITUNGAN, ✂️ CUTTING LIST, 🏗️ CIVIL CALCULATOR, 🔩 MATERIAL, 🔍 VALIDASI, 📊 RINGKASAN, 📝 CATATAN, 🎯 KESIMPULAN, 🔧 TEKNIK LAS.

Gunakan bullet: • Item pertama • Item kedua

Status: ✅ PASS, ❌ FAILED, ⚠️ PERLU DIPERIKSA.

Jangan mengarang ukuran, harga, material, beban, kapasitas, mutu beton, atau spesifikasi yang tidak diberikan. Jika data belum tersedia, tulis "Data belum ditentukan."

Untuk perhitungan: tuliskan asumsi, rumus penting, hitung hasil, validasi ulang, tuliskan hasil akhir, gunakan satuan konsisten.

Jangan menyatakan struktur aman hanya berdasarkan perkiraan sederhana. Untuk struktur yang memerlukan desain, hasil adalah estimasi awal dan harus diverifikasi engineer/insinyur struktur.

Untuk cutting list: hitung semua kebutuhan, hitung batas bawah teoritis, lakukan packing, setiap batang tidak boleh melebihi kapasitas, validasi jumlah potongan, jumlah batang, total material, total sisa, bedakan TRUE WASTE dan REUSABLE OFFCUT, periksa double counting, jika komponen lebih panjang dari batang standar gunakan sambungan dan tandai. Jangan gunakan ceil(total / panjang batang) sebagai hasil final.

JANGAN PERNAH menampilkan API key, token, password, secret.
"""

memory = {}
MAX_MEMORY = 20
GITHUB_API = "https://api.github.com"
GITHUB_MEMORY_DIR = "memory"
MAX_CONTEXT_TURNS = 8
MAX_CONTEXT_CHARS_PER_ITEM = 1200
GROQ_MAX_CONTEXT_TURNS = 4
GROQ_MAX_CONTEXT_CHARS_PER_ITEM = 400
GROQ_MAX_OUTPUT_TOKENS = 1536
OPENROUTER_MAX_OUTPUT_TOKENS = 2048

def history(uid): return memory.setdefault(uid, [])

def remember(uid, role, content):
    history(uid).append({"role": role, "content": content})
    memory[uid] = history(uid)[-MAX_MEMORY:]

def _trim_history_for_context(uid, max_turns=MAX_CONTEXT_TURNS, max_chars_per_item=MAX_CONTEXT_CHARS_PER_ITEM):
    items = history(uid)[-max_turns:]
    trimmed = []
    for m in items:
        content = m.get("content", "") or ""
        if len(content) > max_chars_per_item:
            content = content[:max_chars_per_item] + "\n...(riwayat dipotong untuk hemat token)"
        trimmed.append({"role": m.get("role", "user"), "content": content})
    return trimmed

def _memory_path(uid): return f"{GITHUB_MEMORY_DIR}/{str(uid)}.json"

async def load_persistent_memory(uid):
    uid = str(uid)
    if not GITHUB_TOKEN or not GITHUB_REPO:
        memory.setdefault(uid, []); return
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{_memory_path(uid)}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
        if response.status_code == 404:
            memory[uid] = []; return
        response.raise_for_status()
        encoded = response.json().get("content", "")
        if not encoded: memory[uid] = []; return
        raw = base64.b64decode(encoded.replace("\n", "")).decode("utf-8")
        saved = json.loads(raw)
        if isinstance(saved, dict): saved = saved.get("memory", [])
        if not isinstance(saved, list): saved = []
        memory[uid] = saved[-MAX_MEMORY:]
        log.info("PERSISTENT MEMORY LOAD OK | uid=%s | items=%s", uid, len(memory[uid]))
    except Exception as e:
        log.warning("PERSISTENT MEMORY LOAD FAILED | uid=%s | %s", uid, str(e)[:300])
        memory.setdefault(uid, [])

async def save_persistent_memory(uid):
    uid = str(uid)
    if not GITHUB_TOKEN or not GITHUB_REPO: return
    raw = json.dumps({"user_id": uid, "memory": history(uid)[-MAX_MEMORY:], "updated_at": int(time.time())}, ensure_ascii=False, indent=2)
    encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{_memory_path(uid)}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            current = await client.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
            body = {"message": f"memory: update {uid}", "content": encoded, "branch": GITHUB_BRANCH}
            if current.status_code == 200: body["sha"] = current.json().get("sha")
            elif current.status_code != 404: current.raise_for_status()
            saved = await client.put(url, headers=headers, json=body)
        if saved.status_code not in (200, 201): saved.raise_for_status()
        log.info("PERSISTENT MEMORY SAVE OK | uid=%s", uid)
    except Exception as e:
        log.warning("PERSISTENT MEMORY SAVE FAILED | uid=%s | %s", uid, str(e)[:300])

def build_messages(uid, text, task, max_turns=MAX_CONTEXT_TURNS, max_chars_per_item=MAX_CONTEXT_CHARS_PER_ITEM):
    task_hint = {
        "coding": "TUGAS CODING. Prioritaskan kode yang dapat dijalankan, ketepatan sintaks, debugging, struktur program, solusi praktis.",
        "reasoning": "TUGAS REASONING. Analisis masalah secara sistematis. Periksa kemungkinan penyebab. Validasi kesimpulan sebelum menjawab.",
        "technical": "TUGAS TEKNIK/MANUFAKTUR. Prioritaskan ukuran, material, rangka, fabrikasi, cutting list, jumlah batang, sambungan, efisiensi material, asumsi teknik, pekerjaan sipil, perhitungan las, berat material. Untuk cutting list WAJIB validasi seluruh angka sebelum menjawab.",
        "civil": "TUGAS CIVIL CALCULATOR. Prioritaskan volume, dimensi, kebutuhan material, semen, pasir, kerikil, air, besi, berat besi, pondasi, dinding, plester, acian, galian, urugan. Jika data tidak diberikan jangan mengarang. Untuk struktur hasil adalah estimasi awal, bukan pengganti desain engineer.",
        "math": "TUGAS MATEMATIKA. Hitung dengan teliti. Tampilkan rumus penting. Gunakan satuan. Periksa kembali hasil.",
        "creative": "TUGAS KREATIF. Buat hasil yang siap digunakan, praktis, menarik, dan sesuai tujuan.",
        "general": "TUGAS UMUM. Jawab langsung, jelas, dan berguna.",
    }.get(task, "")
    return [
        {"role": "system", "content": SYSTEM + "\n\n" + task_hint}
    ] + _trim_history_for_context(uid, max_turns, max_chars_per_item) + [
        {"role": "user", "content": text}
    ]

def parse_number(value):
    if value is None: return None
    value = str(value).strip().replace(" ", "")
    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."): value = value.replace(".", "").replace(",", ".")
        else: value = value.replace(",", "")
    elif "," in value: value = value.replace(",", ".")
    try: return float(value)
    except: return None

def fmt(value, decimals=3):
    if value is None: return "-"
    if abs(value - round(value)) < 0.000001: return str(int(round(value)))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

def extract_dimensions(text):
    t = text.lower()
    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)",
    ]
    for pattern in patterns:
        m = re.search(pattern, t)
        if m: return [parse_number(x) for x in m.groups()]
    return []

def extract_value_with_unit(text, units):
    unit_pattern = "|".join(re.escape(x) for x in units)
    pattern = r"(\d+(?:[.,]\d+)?)\s*(" + unit_pattern + r")"
    m = re.search(pattern, text.lower())
    if not m: return None
    value = parse_number(m.group(1)); unit = m.group(2)
    if value is None: return None
    return value, unit

def to_meter(value, unit):
    unit = unit.lower()
    if unit in ("mm",): return value / 1000
    if unit in ("cm",): return value / 100
    if unit in ("m", "meter", "meters"): return value
    return value

# ============================================================
# KALKULATOR TEKNIK LAS
# ============================================================

def hollow_weight(text):
    t = text.lower(); dims = extract_dimensions(text)
    if len(dims) < 3: return None
    w, h, th = dims[0]/1000, dims[1]/1000, dims[2]/1000
    lm = re.search(r"(\d+(?:[.,]\d+)?)\s*m", t)
    if not lm: return None
    length = parse_number(lm.group(1))
    perim = 2*(w+h) - 4*th
    area = perim * th
    wpm = area * 7850
    total = wpm * length
    return f"🔧 BERAT BESI HOLLOW\n\n📋 DATA\n• Ukuran: {fmt(w*1000)} × {fmt(h*1000)} mm\n• Tebal: {fmt(th*1000)} mm\n• Panjang: {fmt(length)} m\n\n🧮 PERHITUNGAN\n• Luas penampang: {fmt(area*1e6)} mm²\n• Berat per meter: {fmt(wpm)} kg/m\n• Berat total: {fmt(wpm)} × {fmt(length)} = {fmt(total)} kg\n\n📊 RINGKASAN\n• Hollow {fmt(w*1000)}×{fmt(h*1000)}×{fmt(th*1000)} mm\n• Panjang: {fmt(length)} m\n• Berat: {fmt(total)} kg\n\n📝 CATATAN\nHasil estimasi, toleransi pabrik ±5%."

def pipe_weight(text):
    t = text.lower()
    dm = re.search(r"(?:pipa|diameter|dia|Ø|ø)\s*(\d+(?:[.,]\d+)?)\s*(mm|inch|\")", t)
    if not dm: return None
    dia = parse_number(dm.group(1)); unit = dm.group(2)
    if unit in ("inch", '"'): dia *= 25.4
    tm = re.search(r"(?:tebal|t)\s*(\d+(?:[.,]\d+)?)\s*mm", t)
    if not tm: return None
    th = parse_number(tm.group(1))
    lm = re.search(r"(\d+(?:[.,]\d+)?)\s*m", t)
    if not lm: return None
    length = parse_number(lm.group(1))
    d, t = dia/1000, th/1000
    wpm = 3.14159 * (d - t) * t * 7850
    total = wpm * length
    return f"🔧 BERAT PIPA BULAT\n\n📋 DATA\n• Diameter: {fmt(dia)} mm ({fmt(dia/25.4, 2)} inch)\n• Tebal: {fmt(th)} mm\n• Panjang: {fmt(length)} m\n\n🧮 PERHITUNGAN\nRumus: π × (D - t) × t × ρ\n• Berat per meter: {fmt(wpm)} kg/m\n• Berat total: {fmt(total)} kg\n\n📊 RINGKASAN\n• Pipa Ø{fmt(dia)} mm × {fmt(th)} mm\n• Panjang: {fmt(length)} m\n• Berat: {fmt(total)} kg\n\n📝 CATATAN\nDiameter adalah diameter luar (OD)."

def plate_weight(text):
    dims = extract_dimensions(text)
    if len(dims) < 3: return None
    p, l, t = dims[0]/1000, dims[1]/1000, dims[2]/1000
    vol = p*l*t; weight = vol*7850
    return f"🔧 BERAT PLAT BESI\n\n📋 DATA\n• Panjang: {fmt(dims[0])} mm\n• Lebar: {fmt(dims[1])} mm\n• Tebal: {fmt(dims[2])} mm\n\n🧮 PERHITUNGAN\nVolume = {fmt(p)} × {fmt(l)} × {fmt(t)} = {fmt(vol)} m³\nBerat = {fmt(vol)} × 7850 = {fmt(weight)} kg\n\n📊 RINGKASAN\n• Plat {fmt(dims[0])}×{fmt(dims[1])}×{fmt(dims[2])} mm\n• Berat: {fmt(weight)} kg\n\n📝 CATATAN\nPlat 1mm = 7.85 kg/m², 3mm = 23.55 kg/m², 5mm = 39.25 kg/m²."

def electrode(text):
    t = text.lower()
    tm = re.search(r"(?:tebal|t)\s*(\d+(?:[.,]\d+)?)\s*mm", t)
    if not tm: return None
    th = parse_number(tm.group(1))
    lm = re.search(r"(?:panjang las|panjang|total)\s*(\d+(?:[.,]\d+)?)\s*m", t)
    if not lm: return None
    length = parse_number(lm.group(1))
    joint = "fillet"
    if "v" in t or "butt" in t or "kampuh" in t: joint = "v-groove"
    epm = th*0.6 if joint == "fillet" else th*0.4
    total = epm*length
    rods = math.ceil(total*5)
    return f"🔧 KEBUTUHAN ELEKTRODA LAS\n\n📋 DATA\n• Tebal plat: {fmt(th)} mm\n• Panjang las: {fmt(length)} m\n• Jenis sambungan: {joint.title()}\n\n🧮 PERHITUNGAN\n• Konsumsi: {fmt(epm)} kg/m\n• Total: {fmt(epm)} × {fmt(length)} = {fmt(total)} kg\n• Estimasi batang 3.2mm: {rods} batang\n\n📊 RINGKASAN\n• Total elektroda: {fmt(total)} kg\n• Batang 3.2mm: {rods} batang\n\n📝 CATATAN\nBeli 10-20% lebih untuk cadangan."

def converter(text):
    t = text.lower()
    if "mm ke m" in t or "mm to m" in t:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*mm", t)
        if m: v = parse_number(m.group(1)); return f"✅ {fmt(v)} mm = {fmt(v/1000)} m"
    if "cm ke m" in t or "cm to m" in t:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*cm", t)
        if m: v = parse_number(m.group(1)); return f"✅ {fmt(v)} cm = {fmt(v/100)} m"
    if "inch ke mm" in t or "inchi ke mm" in t:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:inch|inchi|\")", t)
        if m: v = parse_number(m.group(1)); return f"✅ {fmt(v)} inch = {fmt(v*25.4)} mm"
    if "mm ke inch" in t:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*mm", t)
        if m: v = parse_number(m.group(1)); return f"✅ {fmt(v)} mm = {fmt(v/25.4, 2)} inch"
    if "kg ke ton" in t:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*kg", t)
        if m: v = parse_number(m.group(1)); return f"✅ {fmt(v)} kg = {fmt(v/1000, 3)} ton"
    if "ton ke kg" in t:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*ton", t)
        if m: v = parse_number(m.group(1)); return f"✅ {fmt(v)} ton = {fmt(v*1000)} kg"
    return None

def technical_calculator(text):
    t = text.lower().strip()
    if "hollow" in t or "kotak" in t:
        r = hollow_weight(text)
        if r: return r
    if "pipa" in t:
        r = pipe_weight(text)
        if r: return r
    if "plat" in t and ("berat" in t or "kg" in t):
        r = plate_weight(text)
        if r: return r
    if "elektroda" in t or "kawat las" in t:
        r = electrode(text)
        if r: return r
    if "konversi" in t or "convert" in t:
        r = converter(text)
        if r: return r
    return None

# ============================================================
# CIVIL CALCULATOR
# ============================================================

def concrete_materials(volume, mix=(1,2,3), dry_factor=1.54, cement_density=1440, sack_weight=50, wc=0.50):
    a,b,c = mix
    total_parts = a+b+c
    dry_volume = volume * dry_factor
    cement_volume = dry_volume * a / total_parts
    sand_volume = dry_volume * b / total_parts
    gravel_volume = dry_volume * c / total_parts
    cement_kg = cement_volume * cement_density
    cement_sacks = math.ceil(cement_kg / sack_weight)
    water_liter = cement_kg * wc
    return {"volume": volume, "cement_kg": cement_kg, "cement_sacks": cement_sacks, "sand_m3": sand_volume, "gravel_m3": gravel_volume, "water_liter": water_liter}

def civil_concrete(text, title="KEBUTUHAN BETON"):
    dims = extract_dimensions(text)
    if len(dims) != 3: return None
    p,l,t = dims
    volume = p*l*t
    r = concrete_materials(volume)
    return f"🏗️ {title}\n\n📋 DATA\n• Panjang: {fmt(p)} m\n• Lebar: {fmt(l)} m\n• Tebal: {fmt(t)} m\n• Volume: {fmt(volume)} m³\n\n⚙️ ASUMSI\n• Campuran 1:2:3, faktor kering 1.54\n• Semen 1440 kg/m³, 1 zak = 50 kg\n\n🧮 PERHITUNGAN\n• Semen: {r['cement_sacks']} zak ({fmt(r['cement_kg'])} kg)\n• Pasir: {fmt(r['sand_m3'])} m³\n• Kerikil: {fmt(r['gravel_m3'])} m³\n• Air: {fmt(r['water_liter'])} liter\n\n📝 CATATAN\nHasil estimasi, bukan mix design laboratorium. Verifikasi engineer untuk struktur penting."

def civil_wall(text):
    dims = extract_dimensions(text)
    if len(dims) < 2: return None
    p,l = dims[0], dims[1]
    area = p*l
    if "batako" in text.lower(): material = "batako"; pcs = math.ceil(area*12.5)
    else: material = "bata"; pcs = math.ceil(area*50)
    mortar = area * 0.02
    dry = mortar * 1.33
    cement_kg = dry / 5 * 1440
    sacks = math.ceil(cement_kg / 50)
    sand = dry * 4/5
    return f"🏗️ KEBUTUHAN DINDING\n\n📋 DATA\n• Panjang: {fmt(p)} m\n• Tinggi: {fmt(l)} m\n• Luas: {fmt(area)} m²\n• Material: {material}\n\n🧮 PERHITUNGAN\n• {material}: {pcs} buah\n• Mortar: {fmt(mortar)} m³\n• Semen: {sacks} zak ({fmt(cement_kg)} kg)\n• Pasir: {fmt(sand)} m³\n\n📝 CATATAN\nEstimasi, tergantung ukuran material & metode pemasangan."

def civil_rebar(text):
    t = text.lower()
    dm = re.search(r"(?:d|dia|diameter|besi)\s*(\d+(?:[.,]\d+)?)\s*mm", t)
    if not dm: dm = re.search(r"(\d+(?:[.,]\d+)?)\s*mm", t)
    if not dm: return None
    dia = parse_number(dm.group(1))
    bm = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:batang|btg)", t)
    jml = int(parse_number(bm.group(1))) if bm else None
    pm = re.search(r"(?:panjang|total)\s*(\d+(?:[.,]\d+)?)\s*m", t)
    if not pm:
        sm = re.search(r"(\d+(?:[.,]\d+)?)\s*m", t)
        if sm: pm = sm
    if not pm: return None
    panjang = parse_number(pm.group(1))
    std = 12
    wpm = dia**2/162
    if jml: panjang = jml*std
    total = panjang*wpm
    if not jml: jml = math.ceil(panjang/std)
    return f"🔩 KEBUTUHAN BESI\n\n📋 DATA\n• Diameter: D{fmt(dia)} mm\n• Panjang total: {fmt(panjang)} m\n• Batang standar: {fmt(std)} m\n• Jumlah batang: {jml}\n\n🧮 PERHITUNGAN\n• Berat per meter: {fmt(wpm)} kg/m\n• Berat total: {fmt(total)} kg\n\n📝 CATATAN\nBukan desain tulangan struktur. Verifikasi engineer diperlukan."

def civil_calculator(text):
    t = text.lower().strip()
    if "besi" in t or re.search(r"\bd\d+\b", t) or "diameter" in t:
        r = civil_rebar(text)
        if r: return r
    if "footplat" in t or "foot plate" in t or "tapak" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 3:
            p,l,t = dims[:3]
            vol = p*l*t
            qm = re.search(r"(\d+)\s*(?:buah|bh|unit)", t)
            qty = int(qm.group(1)) if qm else 1
            total = vol*qty
            r = concrete_materials(total)
            return f"🏗️ KEBUTUHAN FOOTPLAT\n\n📋 DATA\n• {fmt(p)} × {fmt(l)} × {fmt(t)} m\n• Jumlah: {qty} buah\n• Total: {fmt(total)} m³\n\n🧮 MATERIAL\n• Semen: {r['cement_sacks']} zak\n• Pasir: {fmt(r['sand_m3'])} m³\n• Kerikil: {fmt(r['gravel_m3'])} m³\n\n⚠️ CATATAN\nBukan desain struktur. Verifikasi engineer diperlukan."
    if "sloof" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 3:
            a,b,length = dims[:3]
            qm = re.search(r"(\d+)\s*(?:buah|bh|unit)", t)
            qty = int(qm.group(1)) if qm else 1
            vol = a*b*length*qty
            r = concrete_materials(vol)
            return f"🏗️ KEBUTUHAN SLOOF\n\n📋 DATA\n• {fmt(a)} × {fmt(b)} m\n• Panjang: {fmt(length)} m\n• Jumlah: {qty}\n• Volume: {fmt(vol)} m³\n\n🧮 MATERIAL\n• Semen: {r['cement_sacks']} zak\n• Pasir: {fmt(r['sand_m3'])} m³\n• Kerikil: {fmt(r['gravel_m3'])} m³\n\n⚠️ CATATAN\nBukan desain struktur. Verifikasi engineer diperlukan."
    if "kolom" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 3:
            a,b,length = dims[:3]
            qm = re.search(r"(\d+)\s*(?:buah|bh|unit)", t)
            qty = int(qm.group(1)) if qm else 1
            vol = a*b*length*qty
            r = concrete_materials(vol)
            return f"🏗️ KEBUTUHAN KOLOM\n\n📋 DATA\n• {fmt(a)} × {fmt(b)} m\n• Tinggi: {fmt(length)} m\n• Jumlah: {qty}\n• Volume: {fmt(vol)} m³\n\n🧮 MATERIAL\n• Semen: {r['cement_sacks']} zak\n• Pasir: {fmt(r['sand_m3'])} m³\n• Kerikil: {fmt(r['gravel_m3'])} m³\n\n⚠️ CATATAN\nBukan desain struktur. Verifikasi engineer diperlukan."
    if "balok" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 3:
            a,b,length = dims[:3]
            qm = re.search(r"(\d+)\s*(?:buah|bh|unit)", t)
            qty = int(qm.group(1)) if qm else 1
            vol = a*b*length*qty
            r = concrete_materials(vol)
            return f"🏗️ KEBUTUHAN BALOK\n\n📋 DATA\n• {fmt(a)} × {fmt(b)} m\n• Panjang: {fmt(length)} m\n• Jumlah: {qty}\n• Volume: {fmt(vol)} m³\n\n🧮 MATERIAL\n• Semen: {r['cement_sacks']} zak\n• Pasir: {fmt(r['sand_m3'])} m³\n• Kerikil: {fmt(r['gravel_m3'])} m³\n\n⚠️ CATATAN\nBukan desain struktur. Verifikasi engineer diperlukan."
    if "plat beton" in t or "plat lantai" in t:
        return civil_concrete(text, "KEBUTUHAN PLAT BETON")
    if "dinding" in t or "tembok" in t or "bata" in t or "batako" in t:
        r = civil_wall(text)
        if r: return r
    if "plester" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 2:
            p,h = dims[0], dims[1]
            area = p*h
            th = 0.015
            td = extract_value_with_unit(text, ["mm","cm","m"])
            if td:
                v,u = td
                cand = to_meter(v,u)
                if 0.003 <= cand <= 0.1: th = cand
            vol = area*th
            dry = vol*1.33
            cem_vol = dry/5
            sand = dry*4/5
            cem_kg = cem_vol*1440
            sacks = math.ceil(cem_kg/50)
            return f"🏗️ KEBUTUHAN PLESTERAN\n\n📋 DATA\n• Panjang: {fmt(p)} m\n• Tinggi: {fmt(h)} m\n• Luas: {fmt(area)} m²\n• Tebal: {fmt(th*1000)} mm\n\n🧮 MATERIAL\n• Semen: {sacks} zak ({fmt(cem_kg)} kg)\n• Pasir: {fmt(sand)} m³\n\n📝 CATATAN\nEstimasi, tergantung ketebalan & kondisi permukaan."
    if "acian" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 2:
            p,h = dims[0], dims[1]
            area = p*h
            th = 2
            td = extract_value_with_unit(text, ["mm","cm"])
            if td:
                v,u = td
                if u == "cm": v *= 10
                if 1 <= v <= 10: th = v
            konsumsi = 1.5*th
            kg = area*konsumsi
            bags = math.ceil(kg/40)
            return f"🏗️ KEBUTUHAN ACIAN\n\n📋 DATA\n• Panjang: {fmt(p)} m\n• Tinggi: {fmt(h)} m\n• Luas: {fmt(area)} m²\n• Tebal: {fmt(th)} mm\n\n🧮 MATERIAL\n• Acian: {fmt(kg)} kg\n• Kemasan 40kg: {bags} zak\n\n📝 CATATAN\nEstimasi, tergantung produk & ketebalan."
    if "galian" in t or "menggali" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 3:
            p,l,t = dims[:3]
            vol = p*l*t
            return f"🏗️ VOLUME GALIAN\n\n📋 DATA\n• Panjang: {fmt(p)} m\n• Lebar: {fmt(l)} m\n• Kedalaman: {fmt(t)} m\n\n🧮 PERHITUNGAN\nVolume = {fmt(p)} × {fmt(l)} × {fmt(t)} = {fmt(vol)} m³\n\n📝 CATATAN\nVolume geometris, belum termasuk faktor pemadatan."
    if "urugan" in t or "urug" in t:
        dims = extract_dimensions(text)
        if len(dims) >= 3:
            p,l,t = dims[:3]
            vol = p*l*t
            return f"🏗️ VOLUME URUGAN\n\n📋 DATA\n• Panjang: {fmt(p)} m\n• Lebar: {fmt(l)} m\n• Tebal: {fmt(t)} m\n\n🧮 PERHITUNGAN\nVolume = {fmt(p)} × {fmt(l)} × {fmt(t)} = {fmt(vol)} m³\n\n📝 CATATAN\nVolume geometris, belum termasuk faktor pemadatan."
    concrete_keywords = ["beton","lantai beton","cor","ngecor","coran","sipil"]
    if any(x in t for x in concrete_keywords):
        r = civil_concrete(text)
        if r: return r
    return None

# ============================================================
# TASK CLASSIFIER
# ============================================================

def classify_task(text):
    t = (text or "").lower()
    coding = ["python", "javascript", "typescript", "php", "html", "css", "sql", "api", "coding", "kode", "program", "programming", "bug", "error", "debug", "github", "vercel", "function", "import ", "async ", "def "]
    civil = ["sipil", "beton", "cor", "coran", "pondasi", "footplat", "sloof", "kolom", "balok", "plat beton", "plat lantai", "dinding", "tembok", "bata", "batako", "plester", "acian", "galian", "urugan", "besi", "besi tulangan", "tulangan", "begel", "bendrat", "berat besi", "diameter besi"]
    technical = ["tenda", "kanopi", "rangka", "hollow", "pipa", "baja", "las", "fabrikasi", "manufaktur", "produksi", "material", "plat", "besi", "aluminium", "konstruksi", "ukuran", "dimensi", "pagar", "bengkel", "welding", "engineering", "cutting list", "potongan batang", "batang 6 meter", "rangka utama", "rangka sekunder", "purlin", "pengaku", "tiang", "balok utama", "sambungan", "elektroda", "kawat las", "berat hollow", "berat pipa", "berat plat"]
    reasoning = ["analisis", "analisa", "kenapa", "mengapa", "bandingkan", "perbandingan", "strategi", "logika", "alasan", "evaluasi", "pecahkan", "solusi terbaik", "reasoning"]
    math_keywords = ["hitung", "perhitungan", "berapa", "rumus", "luas", "volume", "persentase", "matematika", "kg", "meter", "mm", "cm", "m2", "m²"]
    creative = ["caption", "iklan", "promosi", "slogan", "desain", "buatkan gambar", "ide konten", "copywriting"]
    if any(x in t for x in coding): return "coding"
    if any(x in t for x in civil): return "civil"
    if any(x in t for x in technical): return "technical"
    if any(x in t for x in reasoning): return "reasoning"
    if any(x in t for x in math_keywords): return "math"
    if any(x in t for x in creative): return "creative"
    return "general"

# ============================================================
# AI CALLS
# ============================================================

def call_gemini(uid, text, task):
    if not gemini: raise RuntimeError("GEMINI_API_KEY belum tersedia.")
    task_hint = {"coding": "Berikan kode yang dapat dijalankan.", "reasoning": "Analisis masalah secara teliti sebelum kesimpulan.", "technical": "Gunakan pertimbangan teknik dan manufaktur yang praktis.", "civil": "Gunakan perhitungan sipil secara teliti. Jangan mengarang data. Jika data belum diberikan, nyatakan data belum ditentukan. Bedakan estimasi material dengan desain struktur. Untuk struktur jangan menyatakan aman tanpa perhitungan engineering.", "math": "Hitung secara teliti dan tunjukkan asumsi.", "creative": "Buat hasil kreatif yang siap digunakan.", "general": "Jawab langsung dan jelas."}.get(task, "")
    prompt = SYSTEM + "\n\n" + task_hint + "\n\n"
    for m in _trim_history_for_context(uid): prompt += f"{m['role']}: {m['content']}\n"
    prompt += f"user: {text}"
    r = gemini.models.generate_content(model=GEMINI_CHAT_MODEL, contents=prompt)
    answer = r.text or ""
    if not answer.strip(): raise RuntimeError("Gemini mengembalikan jawaban kosong.")
    return answer, GEMINI_CHAT_MODEL

def call_groq(uid, text, task, model=None):
    if not groq: raise RuntimeError("GROQ_API_KEY belum tersedia.")
    if not model:
        if task == "coding": model = GROQ_CODING_MODEL
        elif task in ("reasoning", "math"): model = GROQ_REASONING_MODEL
        else: model = GROQ_FAST_MODEL
    r = groq.chat.completions.create(model=model, messages=build_messages(uid, text, task, max_turns=GROQ_MAX_CONTEXT_TURNS, max_chars_per_item=GROQ_MAX_CONTEXT_CHARS_PER_ITEM), max_tokens=GROQ_MAX_OUTPUT_TOKENS)
    answer = r.choices[0].message.content or ""
    if not answer.strip(): raise RuntimeError(f"Groq ({model}) mengembalikan jawaban kosong.")
    return answer, model

def call_openrouter(uid, text, task, model=None):
    if not openrouter: raise RuntimeError("OPENROUTER_API_KEY belum tersedia.")
    selected = model or OPENROUTER_FREE_MODEL
    r = openrouter.chat.completions.create(model=selected, messages=build_messages(uid, text, task), max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS, extra_headers={"HTTP-Referer": "https://designmanufaktur.vercel.app", "X-Title": "Designmanufaktur Super AI Agent"})
    answer = r.choices[0].message.content or ""
    if not answer.strip(): raise RuntimeError(f"OpenRouter ({selected}) mengembalikan jawaban kosong.")
    selected_model = getattr(r, "model", None) or selected
    return answer, selected_model

def call_cloudflare_chat(uid, text, task, model=None):
    if not cloudflare_text: raise RuntimeError("CLOUDFLARE_API_TOKEN belum tersedia.")
    selected = model or CLOUDFLARE_TEXT_MODEL
    r = cloudflare_text.chat.completions.create(model=selected, messages=build_messages(uid, text, task))
    answer = r.choices[0].message.content or ""
    if not answer.strip(): raise RuntimeError(f"Cloudflare ({selected}) mengembalikan jawaban kosong.")
    return answer, selected

def call_huggingface(uid, text, task, model=None):
    if not huggingface: raise RuntimeError("HUGGINGFACE_API_KEY belum tersedia.")
    selected = model or HUGGINGFACE_MODEL
    r = huggingface.chat.completions.create(model=selected, messages=build_messages(uid, text, task))
    answer = r.choices[0].message.content or ""
    if not answer.strip(): raise RuntimeError(f"HuggingFace ({selected}) mengembalikan jawaban kosong.")
    return answer, selected

def _is_retryable_rate_limit(error_text):
    t = error_text.lower()
    if "per-day" in t or "per day" in t or "daily" in t: return False
    if "429" in t or "rate limit" in t or "resource_exhausted" in t: return True
    return False

def _call_with_retry(fn, retries=1, base_delay=1.5):
    last_err = None
    for attempt in range(retries + 1):
        try: return fn()
        except Exception as e:
            last_err = e
            if attempt == retries or not _is_retryable_rate_limit(str(e)): raise
            time.sleep(base_delay + random.uniform(0, 1.0))
    raise last_err

# ============================================================
# CHAT ROUTER
# ============================================================

def chat_router(uid, text):
    task = classify_task(text)
    log.info("TASK=%s | text=%s", task, text[:120])
    
    # Civil Calculator
    if task == "civil":
        civil_result = civil_calculator(text)
        if civil_result:
            return (civil_result, "Civil Calculator", "Local Calculation Engine", task)
    
    # Technical Calculator
    if task == "technical":
        tech_result = technical_calculator(text)
        if tech_result:
            return (tech_result, "Technical Calculator", "Local Calculation Engine", task)
    
    # Provider Routing
    providers = []
    if task == "coding":
        providers = [("Groq (qwen3-32b)", lambda: call_groq(uid, text, task, model=GROQ_CODING_MODEL)), ("Groq (gpt-oss-120b)", lambda: call_groq(uid, text, task, model=GROQ_REASONING_MODEL)), ("Groq (gpt-oss-20b)", lambda: call_groq(uid, text, task, model=GROQ_FAST_MODEL)), ("OpenRouter Free", lambda: call_openrouter(uid, text, task)), ("Gemini", lambda: call_gemini(uid, text, task)), ("Cloudflare", lambda: call_cloudflare_chat(uid, text, task)), ("HuggingFace", lambda: call_huggingface(uid, text, task))]
    elif task in ("reasoning", "math"):
        providers = [("Groq (gpt-oss-120b)", lambda: call_groq(uid, text, task, model=GROQ_REASONING_MODEL)), ("Groq (gpt-oss-20b)", lambda: call_groq(uid, text, task, model=GROQ_FAST_MODEL)), ("OpenRouter Free", lambda: call_openrouter(uid, text, task)), ("Gemini", lambda: call_gemini(uid, text, task)), ("Cloudflare", lambda: call_cloudflare_chat(uid, text, task)), ("HuggingFace", lambda: call_huggingface(uid, text, task))]
    else:
        providers = [("Groq (gpt-oss-20b)", lambda: call_groq(uid, text, task, model=GROQ_FAST_MODEL)), ("Groq (gpt-oss-120b)", lambda: call_groq(uid, text, task, model=GROQ_REASONING_MODEL)), ("OpenRouter Free", lambda: call_openrouter(uid, text, task)), ("Gemini", lambda: call_gemini(uid, text, task)), ("Cloudflare", lambda: call_cloudflare_chat(uid, text, task)), ("HuggingFace", lambda: call_huggingface(uid, text, task))]
    
    errors = []
    for provider_name, fn in providers:
        try:
            log.info("TRY PROVIDER | task=%s | provider=%s", task, provider_name)
            answer, model = _call_with_retry(fn, retries=1)
            if not answer.strip(): raise RuntimeError("Provider mengembalikan jawaban kosong.")
            log.info("CHAT SUCCESS | task=%s | provider=%s | model=%s", task, provider_name, model)
            return (answer, provider_name, model, task)
        except Exception as e:
            error_text = str(e)
            errors.append(f"{provider_name}: {error_text[:300]}")
            log.warning("PROVIDER FAILED | provider=%s | error=%s", provider_name, error_text[:300])
    raise RuntimeError("Semua provider AI GRATIS gagal. " + " | ".join(errors))

# ============================================================
# TELEGRAM API
# ============================================================

async def tg(method, data):
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN belum diatur.")
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}", json=data)
        r.raise_for_status()
        result = r.json()
    if not result.get("ok"): raise RuntimeError(str(result))
    return result

async def tg_file(file_id):
    result = await tg("getFile", {"file_id": file_id})
    path = result["result"]["file_path"]
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}")
        r.raise_for_status()
    return r.content, path

# ============================================================
# SEND TEXT
# ============================================================

async def send_text(chat_id, text):
    if not text:
        return
    text = str(text).replace("\r\n", "\n")
    text = re.sub(r"```[a-zA-Z0-9_+\-]*\n?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    text = re.sub(r"(?<!\w)\*(?!\s)", "", text)
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = re.sub(r"^\s*[-_*]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("DATA", "📋 DATA")
    text = text.replace("ASUMSI", "⚙️ ASUMSI")
    text = text.replace("PERHITUNGAN", "🧮 PERHITUNGAN")
    text = text.replace("CUTTING LIST", "✂️ CUTTING LIST")
    text = text.replace("VALIDASI", "🔍 VALIDASI")
    text = text.replace("RINGKASAN", "📊 RINGKASAN")
    text = text.replace("CATATAN", "📝 CATATAN")
    text = text.replace("KESIMPULAN", "🎯 KESIMPULAN")
    text = text.replace("TEKNIK LAS", "🔧 TEKNIK LAS")
    text = text.replace("CIVIL CALCULATOR", "🏗️ CIVIL CALCULATOR")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    chunks = []
    current = ""
    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph: continue
        candidate = current + ("\n\n" if current else "") + paragraph
        if len(candidate) <= 3900:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
                current = ""
            lines = paragraph.split("\n")
            for line in lines:
                line = line.strip()
                if not line: continue
                candidate = current + ("\n" if current else "") + line
                if len(candidate) <= 3900:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                        current = ""
                    while len(line) > 3900:
                        chunks.append(line[:3900])
                        line = line[3900:]
                    current = line
    if current:
        chunks.append(current.strip())
    
    for chunk in chunks:
        await tg("sendMessage", {"chat_id": chat_id, "text": chunk})
        if len(chunks) > 1:
            await asyncio.sleep(0.25)

# ============================================================
# SEND PHOTO / VIDEO
# ============================================================

async def send_photo(chat_id, data, filename="image.png", content_type="image/png"):
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": str(chat_id)}, files={"photo": (filename, data, content_type)})
        r.raise_for_status()

async def send_video(chat_id, data):
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo", data={"chat_id": str(chat_id)}, files={"video": ("video.mp4", data, "video/mp4")})
        r.raise_for_status()

# ============================================================
# HANDLE UPDATE
# ============================================================

async def handle(update):
    message = update.get("message")
    if not message: return
    chat_id = message.get("chat", {}).get("id")
    uid = str(message.get("from", {}).get("id", chat_id))
    text = message.get("text", "") or ""
    caption = message.get("caption", "") or ""
    
    if text.startswith("/start"):
        await send_text(chat_id, "🤖 Designmanufaktur Super AI Agent aktif.\n\n🧠 Smart Multi-AI Router\n🏗️ Civil Calculator\n🔧 Teknik Las Calculator\n🖼️ Gemini Vision\n🎨 Free Image Generation\n\nContoh:\n• sipil beton 5 x 10 meter tebal 10 cm\n• berat hollow 40x40x2 mm panjang 6 meter\n• pagar 10 meter tinggi 1.5 meter\n\nPerintah:\n/model\n/reset\n/gambar <prompt>\n\n⚠️ Untuk struktur: hasil material bukan pengganti desain engineer.")
        return
    
    if text.startswith("/reset"):
        memory.pop(uid, None); memory[uid] = []
        await save_persistent_memory(uid)
        await send_text(chat_id, "✅ Memory sesi dihapus.")
        return
    
    if text.startswith("/model"):
        await send_text(chat_id, f"🤖 STATUS SUPER AI AGENT\n\nGemini: {'✅ AKTIF' if gemini else '❌ TIDAK AKTIF'}\nOpenRouter FREE: {'✅ AKTIF' if openrouter else '❌ TIDAK AKTIF'}\nGroq FREE: {'✅ AKTIF' if groq else '❌ TIDAK AKTIF'}\nCloudflare: {'✅ AKTIF' if cloudflare_text else '❌ TIDAK AKTIF'}\nHuggingFace: {'✅ AKTIF' if huggingface else '❌ TIDAK AKTIF (opsional)'}\n\n🧠 MODEL\nGemini: {GEMINI_CHAT_MODEL}\nOpenRouter: {OPENROUTER_FREE_MODEL}\nGroq Coding: {GROQ_CODING_MODEL}\nGroq Reasoning: {GROQ_REASONING_MODEL}\nGroq Fast: {GROQ_FAST_MODEL}")
        return
    
    if text.startswith("/gambar"):
        prompt = command_arg(text)
        if not prompt:
            await send_text(chat_id, "🎨 GENERATE GAMBAR\n\nContoh:\n/gambar pagar minimalis hitam modern")
            return
        await send_text(chat_id, "🎨 Memilih generator gambar GRATIS...")
        try:
            data, provider = await asyncio.to_thread(generate_image, prompt)
            if provider.startswith("Cloudflare"): await send_photo(chat_id, data, filename="image.jpg", content_type="image/jpeg")
            else: await send_photo(chat_id, data)
            await send_text(chat_id, f"✅ Gambar dibuat oleh {provider}.")
        except Exception as e:
            await send_text(chat_id, "❌ Generate gambar gagal.\n" + str(e)[:700])
        return
    
    if message.get("video"):
        await send_text(chat_id, "🎥 Sedang menganalisis video...")
        try:
            data, path = await tg_file(message["video"]["file_id"])
            if len(data) > (20 * 1024 * 1024): await send_text(chat_id, "❌ Video lebih dari 20 MB."); return
            mime = "video/quicktime" if path.lower().endswith(".mov") else "video/mp4"
            answer = await asyncio.to_thread(analyze_video, data, mime, caption or "Analisa video ini secara detail. Jika terkait pekerjaan manufaktur, bengkel, konstruksi, sipil, tenda, kanopi, atau fabrikasi: jelaskan objek, proses, kondisi, masalah, berikan saran praktis. Jangan mengarang ukuran yang tidak terlihat.")
            await send_text(chat_id, answer)
        except Exception as e:
            await send_text(chat_id, "❌ Analisis video gagal.\n" + str(e)[:700])
        return
    
    if message.get("photo"):
        await send_text(chat_id, "🖼️ Gemini Vision sedang menganalisis gambar...")
        try:
            data, path = await tg_file(message["photo"][-1]["file_id"])
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            prompt = caption or "Analisa gambar ini secara detail. Jika terkait manufaktur, bengkel las, tenda, pagar, fabrikasi, konstruksi, sipil, atau produk custom: jelaskan objek, komponen, fungsi, kondisi, masalah, berikan saran praktis. Jangan mengarang ukuran yang tidak terlihat."
            answer, model = await asyncio.to_thread(analyze_image, data, mime, prompt)
            await send_text(chat_id, answer)
            log.info("VISION SUCCESS | model=%s", model)
        except Exception as e:
            await send_text(chat_id, "❌ Analisis gambar gagal.\n" + str(e)[:700])
        return
    
    if not text: return
    try:
        await load_persistent_memory(uid)
        await tg("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        answer, provider, model, task = await asyncio.to_thread(chat_router, uid, text)
        remember(uid, "user", text)
        remember(uid, "assistant", answer)
        await save_persistent_memory(uid)
        await send_text(chat_id, answer)
        log.info("CHAT DONE | task=%s | provider=%s | model=%s", task, provider, model)
    except Exception as e:
        log.exception("chat failed")
        await send_text(chat_id, "❌ Semua AI GRATIS gagal untuk request ini.\n\n" + str(e)[:700])

# ============================================================
# IMAGE GENERATION
# ============================================================

def pollinations_image(prompt):
    if not POLLINATIONS_ENABLED: raise RuntimeError("Pollinations tidak diaktifkan.")
    if not POLLINATIONS_KEY: raise RuntimeError("POLLINATIONS_API_KEY belum tersedia.")
    from urllib.parse import quote
    url = f"{POLLINATIONS_BASE_URL}/image/{quote(prompt, safe='')}?model={quote(POLLINATIONS_IMAGE_MODEL)}&width=1024&height=1024"
    with httpx.Client(timeout=300) as client:
        r = client.get(url, headers={"Authorization": f"Bearer {POLLINATIONS_KEY}", "Accept": "image/png,image/jpeg,*/*"})
        if r.status_code >= 400: raise RuntimeError(f"Pollinations HTTP {r.status_code}: {r.text[:400]}")
        if not r.content: raise RuntimeError("Pollinations mengembalikan data kosong.")
        return r.content

def _to_jpeg_bytes(raw):
    try:
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(raw))
        if img.mode in ("RGBA", "P", "LA"): img = img.convert("RGB")
        out = _io.BytesIO()
        img.save(out, format="JPEG", quality=92)
        return out.getvalue()
    except: return raw

def cloudflare_flux_image(prompt):
    if not CLOUDFLARE_ENABLED: raise RuntimeError("Cloudflare belum dikonfigurasi.")
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    with httpx.Client(timeout=300) as client:
        r = client.post(url, headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}", "Content-Type": "application/json"}, json={"prompt": prompt})
        if r.status_code >= 400: raise RuntimeError(f"Cloudflare HTTP {r.status_code}: {r.text[:400]}")
        content_type = r.headers.get("content-type", "")
        if content_type.startswith("image/"): raw = r.content
        else:
            payload = r.json()
            if not payload.get("success", True): raise RuntimeError(f"Cloudflare gagal: {payload.get('errors')}")
            result = payload.get("result") or {}
            b64 = result.get("image")
            if not b64: raise RuntimeError("Cloudflare mengembalikan data tidak dikenal.")
            raw = base64.b64decode(b64)
        if not raw: raise RuntimeError("Cloudflare mengembalikan data kosong.")
        return _to_jpeg_bytes(raw)

def generate_image(prompt):
    errors = []
    if CLOUDFLARE_ENABLED:
        try: return (cloudflare_flux_image(prompt), "Cloudflare Workers AI (FLUX)")
        except Exception as e: errors.append(f"Cloudflare: {e}")
    if POLLINATIONS_ENABLED:
        try: return (pollinations_image(prompt), "Pollinations")
        except Exception as e: errors.append(f"Pollinations: {e}")
    raise RuntimeError("Generate gambar GRATIS belum tersedia.\n" + "\n".join(errors))

def command_arg(text):
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""

# ============================================================
# VISION
# ============================================================

def analyze_image(data, mime, prompt):
    if not gemini: raise RuntimeError("Gemini belum dikonfigurasi.")
    errors = []
    try:
        r = gemini.models.generate_content(model=GEMINI_CHAT_MODEL, contents=[types.Part.from_bytes(data=data, mime_type=mime), SYSTEM + "\n\n" + prompt])
        if r.text: return (r.text, GEMINI_CHAT_MODEL)
    except Exception as e: errors.append("Gemini Vision: " + str(e)[:220])
    if openrouter:
        try:
            b64 = base64.b64encode(data).decode()
            content = [{"type": "text", "text": SYSTEM + "\n\n" + prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
            r = openrouter.chat.completions.create(model=OPENROUTER_FREE_MODEL, messages=[{"role": "user", "content": content}], max_tokens=4096)
            answer = r.choices[0].message.content or ""
            if answer.strip(): return (answer, getattr(r, "model", OPENROUTER_FREE_MODEL))
        except Exception as e: errors.append("OpenRouter Vision: " + str(e)[:220])
    raise RuntimeError("Semua provider vision gagal: " + " | ".join(errors))

def analyze_video(data, mime, prompt):
    if not gemini: raise RuntimeError("Gemini diperlukan untuk video.")
    uploaded = gemini.files.upload(file=types.Part.from_bytes(data=data, mime_type=mime))
    for _ in range(60):
        f = gemini.files.get(name=uploaded.name)
        state = getattr(getattr(f, "state", None), "name", "")
        if state == "ACTIVE": uploaded = f; break
        if state == "FAILED": raise RuntimeError("Gemini gagal memproses video.")
        time.sleep(2)
    else: raise RuntimeError("Video belum siap diproses.")
    result = gemini.models.generate_content(model=GEMINI_CHAT_MODEL, contents=[uploaded, SYSTEM + "\n\n" + prompt])
    return result.text or ""

# ============================================================
# WEBHOOK & ROOT
# ============================================================

@app.get("/")
async def root():
    return {"ok": True, "service": "Designmanufaktur Super AI Agent", "free_only": True}

@app.get("/api")
async def api_root(): return await root()

async def webhook_impl(request: Request, x_telegram_bot_api_secret_token: Optional[str]):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    update = await request.json()
    await handle(update)
    return {"ok": True}

@app.post("/api/webhook")
async def webhook(request: Request, x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)):
    return await webhook_impl(request, x_telegram_bot_api_secret_token)

@app.post("/")
async def root_post(request: Request, x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)):
    return await webhook_impl(request, x_telegram_bot_api_secret_token)