"""
Designmanufaktur Super AI Agent Telegram
Vercel / FastAPI / Python

Main features:
- Telegram webhook
- Expert routing: Coding / Reasoning / Manufacturing / Chat
- Gemini + Groq fallback
- GitHub persistent memory
- GitHub knowledge base + retrieval
- GitHub few-shot examples
- TXT/MD/CSV/JSON/PY/JS/TS/HTML/CSS/XML/YAML/YML/PDF/DOCX ingestion
- Photo analysis and video analysis with Gemini
- Optional Pollinations image/video generation
- Commands: /start /model /memory /knowledge /remember /forget /reset /gambar /video

Secrets are read only from environment variables.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from pathlib import PurePosixPath
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from google import genai
from google.genai import types
from openai import OpenAI

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("designmanufaktur")

app = FastAPI(title="Designmanufaktur Super AI Agent")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Accept common human-written values such as "Gemini_2_5_Flash".
def normalize_gemini_model(value: str) -> str:
    value = (value or "").strip().lower()
    aliases = {
        "gemini_2_5_flash": "gemini-2.5-flash",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini_2_5_flash_lite": "gemini-2.5-flash-lite",
        "gemini 2.5 flash lite": "gemini-2.5-flash-lite",
    }
    return aliases.get(value, value)

GEMINI_MODEL = normalize_gemini_model(GEMINI_MODEL)
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]
GEMINI_ACTIVE_MODEL = GEMINI_MODEL

GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_REASONING_MODEL = os.getenv(
    "GROQ_REASONING_MODEL", "openai/gpt-oss-120b"
).strip()
GROQ_CODING_MODEL = os.getenv(
    "GROQ_CODING_MODEL", "qwen/qwen3-32b"
).strip()
GROQ_FAST_MODEL = os.getenv(
    "GROQ_FAST_MODEL", "openai/gpt-oss-20b"
).strip()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.getenv(
    "GITHUB_REPO", "Dinal7297/designmanufaktur-memory"
).strip()
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main").strip()

POLLINATIONS_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()
POLLINATIONS_ENABLED = os.getenv(
    "POLLINATIONS_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
POLLINATIONS_IMAGE_MODEL = os.getenv(
    "POLLINATIONS_IMAGE_MODEL", "flux"
).strip()
POLLINATIONS_VIDEO_MODEL = os.getenv(
    "POLLINATIONS_VIDEO_MODEL", "wan"
).strip()

# Optional image-generation model. If absent, /gambar uses the stable
# Gemini image model as a fallback when GEMINI_API_KEY exists.
GEMINI_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image"
).strip()

# Telegram bot download limit is 20 MB for getFile.
MAX_TELEGRAM_DOWNLOAD = 20 * 1024 * 1024
MAX_HISTORY = 24
MAX_KNOWLEDGE_CHUNKS = 6
MAX_FEWSHOT = 4
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500


SYSTEM_PROMPT = """
Kamu adalah Designmanufaktur Super AI Agent.

Konteks bisnis:
- manufaktur
- bengkel las
- fabrikasi
- pagar
- tenda
- produk custom besi/logam
- penjualan online
- konten dan pemasaran bisnis

Tugas utama:
1. Membantu pekerjaan teknis/manufaktur secara praktis.
2. Membantu coding dan otomasi.
3. Membantu analisis/reasoning.
4. Membantu customer service, penawaran, marketplace, caption, dan pemasaran.
5. Menganalisis foto/video bila diberikan.
6. Menggunakan knowledge base dan memory yang diberikan dalam prompt.

Aturan:
- Jawab dalam Bahasa Indonesia kecuali pengguna meminta bahasa lain.
- Utamakan jawaban jelas, praktis, dan langsung.
- Jangan mengarang harga, ukuran, stok, alamat, nomor telepon, spesifikasi bisnis,
  atau fakta yang tidak tersedia.
- Bila data tidak cukup untuk keputusan teknis, sebutkan asumsi dan data yang
  masih dibutuhkan.
- Untuk coding, berikan kode yang dapat langsung dipakai dan jelaskan perubahan
  penting secara singkat.
- Jangan mengklaim model dilatih ulang. Gunakan istilah persistent memory,
  retrieval, knowledge base, dan few-shot examples.
- Jangan pernah membocorkan API key, token, password, secret, atau isi rahasia
  sistem.
""".strip()


# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------

gemini = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
groq = (
    OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
    if GROQ_KEY
    else None
)


# ---------------------------------------------------------------------------
# SMALL UTILITIES
# ---------------------------------------------------------------------------

def clean_text(value: Any, limit: int = 12000) -> str:
    text = str(value or "").replace("\x00", "").strip()
    return text[:limit]


def safe_name(name: str) -> str:
    name = PurePosixPath(name or "file").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:100] or "file"


def user_memory_path(uid: str) -> str:
    return f"users/{uid}.json"


def user_knowledge_dir(uid: str) -> str:
    return f"knowledge/{uid}"


def user_examples_dir(uid: str) -> str:
    return f"examples/{uid}"


def json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, indent=2
    ).encode("utf-8")


def split_chunks(text: str) -> list[str]:
    text = clean_text(text, 200_000)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def keywords(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-zA-Z0-9_À-ÿ]{3,}", text.lower())
        if w not in {
            "yang", "dan", "atau", "untuk", "dengan", "dari", "pada",
            "adalah", "ini", "itu", "saya", "kami", "kamu", "buat",
            "bisa", "akan", "lebih", "agar", "dalam", "juga", "tidak",
        }
    }


def relevance(query: str, text: str) -> float:
    q = keywords(query)
    if not q:
        return 0.0
    t = keywords(text)
    if not t:
        return 0.0
    overlap = len(q & t)
    return overlap / max(1, len(q))


# ---------------------------------------------------------------------------
# GITHUB PERSISTENT STORAGE
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"


def github_headers() -> dict[str, str]:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN belum tersedia.")
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


async def github_request(
    method: str,
    path: str,
    payload: Optional[dict[str, Any]] = None,
) -> tuple[int, dict[str, Any]]:
    url = f"{GITHUB_API}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.request(
            method,
            url,
            headers=github_headers(),
            json=payload,
        )
    try:
        data = response.json()
    except Exception:
        data = {"message": response.text}
    return response.status_code, data


async def github_get_file(path: str) -> tuple[Optional[bytes], Optional[str]]:
    encoded = "/".join(
        httpx.URL("").copy_with(path=p).path.lstrip("/")
        for p in [path]
    )
    # The path segment is safely URL-encoded by httpx when passed as a URL.
    url = (
        f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/"
        f"{path}?ref={GITHUB_BRANCH}"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=github_headers())

    if response.status_code == 404:
        return None, None
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub GET gagal ({response.status_code}): "
            f"{response.text[:300]}"
        )

    data = response.json()
    content = base64.b64decode(data["content"].replace("\n", ""))
    return content, data.get("sha")


async def github_put_file(
    path: str,
    content: bytes,
    message: str,
) -> None:
    _, sha = await github_get_file(path)
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    encoded_path = path
    url = (
        f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/"
        f"{encoded_path}"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(
            url,
            headers=github_headers(),
            json=payload,
        )
    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"GitHub PUT gagal ({response.status_code}): "
            f"{response.text[:400]}"
        )


async def github_delete_file(path: str, message: str) -> bool:
    _, sha = await github_get_file(path)
    if not sha:
        return False

    payload = {
        "message": message,
        "sha": sha,
        "branch": GITHUB_BRANCH,
    }
    url = (
        f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/"
        f"{path}"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.delete(
            url,
            headers=github_headers(),
            json=payload,
        )
    if response.status_code not in (200, 204):
        raise RuntimeError(
            f"GitHub DELETE gagal ({response.status_code}): "
            f"{response.text[:400]}"
        )
    return True


async def github_list_dir(path: str) -> list[dict[str, Any]]:
    url = (
        f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/"
        f"{path}?ref={GITHUB_BRANCH}"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=github_headers())
    if response.status_code == 404:
        return []
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub LIST gagal ({response.status_code}): "
            f"{response.text[:300]}"
        )
    data = response.json()
    return data if isinstance(data, list) else []


async def load_memory(uid: str) -> dict[str, Any]:
    raw, _ = await github_get_file(user_memory_path(uid))
    if not raw:
        return {
            "version": 1,
            "user_id": uid,
            "facts": [],
            "history": [],
        }
    try:
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError
        obj.setdefault("version", 1)
        obj.setdefault("user_id", uid)
        obj.setdefault("facts", [])
        obj.setdefault("history", [])
        return obj
    except Exception:
        log.exception("Memory JSON rusak untuk user %s", uid)
        return {
            "version": 1,
            "user_id": uid,
            "facts": [],
            "history": [],
        }


async def save_memory(uid: str, mem: dict[str, Any]) -> None:
    mem["history"] = mem.get("history", [])[-MAX_HISTORY:]
    mem["facts"] = mem.get("facts", [])[-100:]
    await github_put_file(
        user_memory_path(uid),
        json_bytes(mem),
        f"Update memory user {uid}",
    )


async def add_history(uid: str, role: str, content: str) -> None:
    mem = await load_memory(uid)
    mem["history"].append({
        "role": role,
        "content": clean_text(content, 12000),
        "ts": int(time.time()),
    })
    await save_memory(uid, mem)


async def remember_fact(uid: str, fact: str) -> None:
    mem = await load_memory(uid)
    fact = clean_text(fact, 2000)
    if fact and fact not in mem["facts"]:
        mem["facts"].append(fact)
    await save_memory(uid, mem)


async def forget_fact(uid: str, query: str) -> int:
    mem = await load_memory(uid)
    old = list(mem.get("facts", []))
    q = query.lower().strip()
    mem["facts"] = [
        f for f in old
        if q not in f.lower()
    ]
    removed = len(old) - len(mem["facts"])
    if removed:
        await save_memory(uid, mem)
    return removed


async def reset_user(uid: str) -> None:
    await github_delete_file(
        user_memory_path(uid),
        f"Reset memory user {uid}",
    )

    for directory in (user_knowledge_dir(uid), user_examples_dir(uid)):
        items = await github_list_dir(directory)
        for item in items:
            if item.get("type") == "file" and item.get("path"):
                try:
                    await github_delete_file(
                        item["path"],
                        f"Reset data user {uid}",
                    )
                except Exception:
                    log.exception("Gagal menghapus %s", item.get("path"))


async def retrieve_knowledge(uid: str, query: str) -> list[str]:
    items = await github_list_dir(user_knowledge_dir(uid))
    scored: list[tuple[float, str]] = []

    for item in items:
        if item.get("type") != "file":
            continue
        path = item.get("path")
        if not path:
            continue
        try:
            raw, _ = await github_get_file(path)
            if not raw:
                continue
            obj = json.loads(raw.decode("utf-8"))
            text = obj.get("text", "")
            score = relevance(query, text)
            if score > 0:
                scored.append((score, text))
        except Exception:
            log.exception("Gagal membaca knowledge %s", path)

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:MAX_KNOWLEDGE_CHUNKS]]


async def retrieve_examples(uid: str, query: str) -> list[dict[str, str]]:
    items = await github_list_dir(user_examples_dir(uid))
    scored: list[tuple[float, dict[str, str]]] = []

    for item in items:
        if item.get("type") != "file":
            continue
        path = item.get("path")
        if not path:
            continue
        try:
            raw, _ = await github_get_file(path)
            if not raw:
                continue
            obj = json.loads(raw.decode("utf-8"))
            q = obj.get("question", "")
            a = obj.get("answer", "")
            score = relevance(query, q + " " + a)
            if score > 0:
                scored.append((score, {
                    "question": q,
                    "answer": a,
                    "category": obj.get("category", "general"),
                }))
        except Exception:
            log.exception("Gagal membaca example %s", path)

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:MAX_FEWSHOT]]


async def save_example(
    uid: str,
    question: str,
    answer: str,
    category: str,
) -> None:
    example_id = uuid.uuid4().hex
    path = f"{user_examples_dir(uid)}/{example_id}.json"
    obj = {
        "version": 1,
        "user_id": uid,
        "question": clean_text(question, 5000),
        "answer": clean_text(answer, 8000),
        "category": category,
        "ts": int(time.time()),
    }
    await github_put_file(
        path,
        json_bytes(obj),
        f"Add few-shot example user {uid}",
    )


# ---------------------------------------------------------------------------
# KNOWLEDGE EXTRACTION
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts",
    ".html", ".css", ".xml", ".yaml", ".yml",
}


def extract_document(data: bytes, filename: str) -> str:
    ext = PurePosixPath(filename.lower()).suffix

    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")

    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("Library pypdf belum tersedia.")
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    if ext == ".docx":
        if Document is None:
            raise RuntimeError("Library python-docx belum tersedia.")
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    raise RuntimeError(
        "Format tidak didukung. Gunakan TXT, MD, CSV, JSON, PY, JS, TS, "
        "HTML, CSS, XML, YAML, YML, PDF, atau DOCX."
    )


async def ingest_knowledge(
    uid: str,
    filename: str,
    data: bytes,
) -> int:
    text = extract_document(data, filename)
    chunks = split_chunks(text)
    if not chunks:
        raise RuntimeError("Dokumen tidak berisi teks yang dapat dipelajari.")

    stem = safe_name(filename)
    total = 0

    for index, chunk in enumerate(chunks):
        path = (
            f"{user_knowledge_dir(uid)}/"
            f"{stem}.{index:04d}.json"
        )
        obj = {
            "version": 1,
            "user_id": uid,
            "source": filename,
            "chunk": index,
            "text": chunk,
            "ts": int(time.time()),
        }
        await github_put_file(
            path,
            json_bytes(obj),
            f"Add knowledge {filename} chunk {index}",
        )
        total += 1

    return total


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

def classify_request(text: str) -> str:
    t = text.lower()

    coding = [
        "python", "javascript", "typescript", "php", "html", "css",
        "sql", "api", "debug", "bug", "error", "coding", "kode",
        "program", "programming", "github", "vercel", "webhook",
        "function", "class", "script", "json",
    ]
    reasoning = [
        "analisis", "analysis", "hitung", "perhitungan", "bandingkan",
        "strategi", "kenapa", "mengapa", "sebab", "logika", "reasoning",
        "evaluasi", "risiko", "optimasi", "optimal", "keputusan",
    ]
    manufacturing = [
        "manufaktur", "fabrikasi", "las", "welding", "pagar", "tenda",
        "besi", "baja", "stainless", "aluminium", "plat", "pipa",
        "rangka", "cutting", "bending", "finishing", "produksi",
        "material", "konstruksi", "ukuran", "ketebalan", "fabrication",
    ]

    if any(k in t for k in coding):
        return "coding"
    if any(k in t for k in reasoning):
        return "reasoning"
    if any(k in t for k in manufacturing):
        return "manufacturing"
    return "chat"


async def build_prompt(uid: str, user_text: str) -> str:
    mem = await load_memory(uid)
    knowledge = await retrieve_knowledge(uid, user_text)
    examples = await retrieve_examples(uid, user_text)

    history = mem.get("history", [])[-MAX_HISTORY:]
    facts = mem.get("facts", [])

    parts = [SYSTEM_PROMPT]

    if facts:
        parts.append(
            "PERSISTENT MEMORY / FAKTA PENGGUNA:\n- "
            + "\n- ".join(clean_text(x, 1000) for x in facts[-50:])
        )

    if history:
        parts.append(
            "RIWAYAT PERCAKAPAN TERBARU:\n"
            + "\n".join(
                f"{m.get('role')}: {clean_text(m.get('content'), 4000)}"
                for m in history
            )
        )

    if knowledge:
        parts.append(
            "KNOWLEDGE BASE RELEVAN:\n"
            + "\n\n---\n\n".join(knowledge)
        )

    if examples:
        parts.append(
            "FEW-SHOT EXAMPLES RELEVAN:\n"
            + "\n\n".join(
                f"Contoh {i+1} ({e.get('category')}):\n"
                f"Q: {e.get('question')}\n"
                f"A: {e.get('answer')}"
                for i, e in enumerate(examples)
            )
        )

    parts.append(f"PESAN PENGGUNA:\n{user_text}\n\nJawab pesan pengguna.")
    return "\n\n".join(parts)


async def _gemini_generate(model: str, contents: Any, config: Any = None) -> Any:
    if not gemini:
        raise RuntimeError("Gemini tidak tersedia.")
    kwargs = {"model": model, "contents": contents}
    if config is not None:
        kwargs["config"] = config
    return await asyncio.to_thread(gemini.models.generate_content, **kwargs)


async def _gemini_call_with_fallback(contents: Any, config: Any = None) -> tuple[Any, str]:
    """Try the configured model first, then safe Gemini fallbacks.

    This protects the bot when a Google API key/project cannot access the
    configured model. The preferred model remains GEMINI_MODEL.
    """
    global GEMINI_ACTIVE_MODEL
    if not gemini:
        raise RuntimeError("Gemini tidak tersedia.")

    candidates = []
    for model in [GEMINI_ACTIVE_MODEL, GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]:
        model = normalize_gemini_model(model)
        if model and model not in candidates:
            candidates.append(model)

    errors = []
    for model in candidates:
        try:
            response = await _gemini_generate(model, contents, config)
            GEMINI_ACTIVE_MODEL = model
            return response, model
        except Exception as exc:
            message = str(exc)
            errors.append(f"{model}: {message[:220]}")
            log.warning("Gemini model %s gagal: %s", model, message[:300])

    raise RuntimeError("Semua model Gemini gagal. " + " | ".join(errors))


async def call_gemini(uid: str, text: str) -> str:
    if not gemini:
        raise RuntimeError("Gemini tidak tersedia.")
    prompt = await build_prompt(uid, text)
    response, _ = await _gemini_call_with_fallback(prompt)
    answer = clean_text(getattr(response, "text", ""), 16000)
    if not answer:
        raise RuntimeError("Gemini tidak mengembalikan jawaban.")
    return answer


async def call_groq(
    uid: str,
    text: str,
    model: str,
) -> str:
    if not groq:
        raise RuntimeError("Groq tidak tersedia.")
    prompt = await build_prompt(uid, text)
    # Keep Groq fallback payload safely bounded; GitHub memory/knowledge can
    # otherwise make the OpenAI-compatible request too large (HTTP 413).
    if len(prompt) > 28000:
        prompt = (
            prompt[:6000]
            + "\n\n[RIWAYAT/KNOWLEDGE DIPANGKAS UNTUK BATAS PAYLOAD]\n\n"
            + prompt[-22000:]
        )

    def _call() -> str:
        response = groq.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return clean_text(
            response.choices[0].message.content
            if response.choices
            else "",
            16000,
        )

    answer = await asyncio.to_thread(_call)
    if not answer:
        raise RuntimeError("Groq tidak mengembalikan jawaban.")
    return answer


async def route_chat(uid: str, text: str) -> tuple[str, str, str]:
    category = classify_request(text)

    if category == "coding":
        candidates = [
            ("Groq Coding", lambda: call_groq(uid, text, GROQ_CODING_MODEL)),
            ("Gemini", lambda: call_gemini(uid, text)),
        ]
    elif category == "reasoning":
        candidates = [
            ("Groq Reasoning", lambda: call_groq(uid, text, GROQ_REASONING_MODEL)),
            ("Gemini", lambda: call_gemini(uid, text)),
        ]
    elif category == "manufacturing":
        candidates = [
            ("Gemini", lambda: call_gemini(uid, text)),
            ("Groq Reasoning", lambda: call_groq(uid, text, GROQ_REASONING_MODEL)),
        ]
    else:
        candidates = [
            ("Gemini", lambda: call_gemini(uid, text)),
            ("Groq Fast", lambda: call_groq(uid, text, GROQ_FAST_MODEL)),
        ]

    errors = []
    for name, fn in candidates:
        try:
            answer = await fn()
            if answer:
                return answer, name, category
        except Exception as exc:
            log.exception("Provider %s gagal", name)
            errors.append(f"{name}: {str(exc)[:180]}")

    raise RuntimeError(
        "Semua provider untuk kategori "
        f"{category} gagal. " + " | ".join(errors)
    )


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------

async def tg(
    method: str,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN belum tersedia.")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
            json=payload or {},
        )

    try:
        data = response.json()
    except Exception:
        data = {}

    if response.status_code >= 400 or not data.get("ok"):
        raise RuntimeError(
            f"Telegram {method} gagal: {response.text[:400]}"
        )
    return data


async def tg_file(file_id: str) -> tuple[bytes, str]:
    data = await tg("getFile", {"file_id": file_id})
    path = data["result"]["file_path"]

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}"
        )
    response.raise_for_status()

    if len(response.content) > MAX_TELEGRAM_DOWNLOAD:
        raise RuntimeError("File lebih besar dari 20 MB.")

    return response.content, path


async def send_text(chat_id: int | str, text: str) -> None:
    text = clean_text(text, 20000) or "Tidak ada jawaban."
    for i in range(0, len(text), 3900):
        await tg(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text[i:i + 3900],
            },
        )


async def send_document(
    chat_id: int | str,
    data: bytes,
    filename: str,
) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument",
            data={"chat_id": str(chat_id)},
            files={
                "document": (
                    safe_name(filename),
                    data,
                    mimetypes.guess_type(filename)[0] or "application/octet-stream",
                )
            },
        )
    response.raise_for_status()


async def send_photo(chat_id: int | str, data: bytes) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data={"chat_id": str(chat_id)},
            files={"photo": ("image.png", data, "image/png")},
        )
    response.raise_for_status()


async def send_video(chat_id: int | str, data: bytes) -> None:
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",
            data={"chat_id": str(chat_id)},
            files={"video": ("video.mp4", data, "video/mp4")},
        )
    response.raise_for_status()


# ---------------------------------------------------------------------------
# MULTIMODAL
# ---------------------------------------------------------------------------

def extract_inline_image(response: Any) -> Optional[bytes]:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
    return None


async def analyze_image(data: bytes, mime: str, prompt: str) -> str:
    if not gemini:
        raise RuntimeError("Gemini diperlukan untuk analisis foto.")

    response, _ = await _gemini_call_with_fallback([
        types.Part.from_bytes(data=data, mime_type=mime),
        SYSTEM_PROMPT + "\n\n" + prompt,
    ])
    answer = clean_text(getattr(response, "text", ""), 16000)
    if not answer:
        raise RuntimeError("Gemini tidak mengembalikan analisis gambar.")
    return answer


async def analyze_video(data: bytes, mime: str, prompt: str) -> str:
    if not gemini:
        raise RuntimeError("Gemini diperlukan untuk analisis video.")

    def _upload():
        return gemini.files.upload(
            file=types.Part.from_bytes(data=data, mime_type=mime)
        )

    uploaded = await asyncio.to_thread(_upload)

    for _ in range(60):
        file_obj = await asyncio.to_thread(
            gemini.files.get,
            name=uploaded.name,
        )
        state = getattr(
            getattr(file_obj, "state", None),
            "name",
            "",
        )
        if state == "ACTIVE":
            uploaded = file_obj
            break
        if state == "FAILED":
            raise RuntimeError("Gemini gagal memproses video.")
        await asyncio.sleep(2)
    else:
        raise RuntimeError("Video terlalu lama diproses Gemini.")

    response, _ = await _gemini_call_with_fallback([
        uploaded,
        SYSTEM_PROMPT + "\n\n" + prompt,
    ])
    answer = clean_text(getattr(response, "text", ""), 16000)
    if not answer:
        raise RuntimeError("Gemini tidak mengembalikan analisis video.")
    return answer


async def gemini_generate_image(prompt: str) -> bytes:
    if not gemini:
        raise RuntimeError("Gemini tidak tersedia.")

    response = await asyncio.to_thread(
        gemini.models.generate_content,
        model=GEMINI_IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        ),
    )
    data = extract_inline_image(response)
    if not data:
        raise RuntimeError(
            "Gemini image tidak mengembalikan data gambar."
        )
    return data


async def pollinations_image(prompt: str) -> bytes:
    if not POLLINATIONS_ENABLED or not POLLINATIONS_KEY:
        raise RuntimeError("Pollinations image tidak aktif.")

    from urllib.parse import quote

    url = (
        "https://gen.pollinations.ai/image/"
        + quote(prompt, safe="")
        + f"?model={quote(POLLINATIONS_IMAGE_MODEL, safe='')}"
    )
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {POLLINATIONS_KEY}"},
        )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("Pollinations tidak mengembalikan gambar.")
    return response.content


async def generate_image(prompt: str) -> tuple[bytes, str]:
    candidates = []

    if POLLINATIONS_ENABLED and POLLINATIONS_KEY:
        candidates.append(("Pollinations", lambda: pollinations_image(prompt)))

    if gemini:
        candidates.append(("Gemini Image", lambda: gemini_generate_image(prompt)))

    errors = []
    for name, fn in candidates:
        try:
            return await fn(), name
        except Exception as exc:
            log.exception("%s image failed", name)
            errors.append(f"{name}: {str(exc)[:180]}")

    raise RuntimeError(
        "Semua provider gambar gagal. " + " | ".join(errors)
    )


async def pollinations_video(prompt: str) -> bytes:
    if not POLLINATIONS_ENABLED or not POLLINATIONS_KEY:
        raise RuntimeError("Pollinations video tidak aktif.")

    from urllib.parse import quote

    url = (
        "https://gen.pollinations.ai/video/"
        + quote(prompt, safe="")
        + f"?model={quote(POLLINATIONS_VIDEO_MODEL, safe='')}"
    )
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {POLLINATIONS_KEY}"},
        )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("Pollinations tidak mengembalikan video.")
    return response.content


async def generate_video(prompt: str) -> tuple[bytes, str]:
    # Per prioritas proyek: Pollinations only when enabled/available.
    if POLLINATIONS_ENABLED and POLLINATIONS_KEY:
        return await pollinations_video(prompt), "Pollinations"

    raise RuntimeError(
        "Generate video tidak aktif. Aktifkan Pollinations dengan "
        "POLLINATIONS_ENABLED=true dan POLLINATIONS_API_KEY."
    )


# ---------------------------------------------------------------------------
# COMMANDS / UPDATE HANDLER
# ---------------------------------------------------------------------------

def command_arg(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


def is_command(text: str, command: str) -> bool:
    return text == command or text.startswith(command + " ")


async def handle_update(update: dict[str, Any]) -> None:
    message = update.get("message")
    if not message:
        return

    chat_id = message.get("chat", {}).get("id")
    if chat_id is None:
        return

    uid = str(message.get("from", {}).get("id", chat_id))
    text = clean_text(message.get("text", ""), 10000)
    caption = clean_text(message.get("caption", ""), 5000)

    # /start
    if is_command(text, "/start"):
        await send_text(
            chat_id,
            "🤖 Designmanufaktur Super AI Agent aktif.\n\n"
            "Routing otomatis:\n"
            "• Coding → Groq Coding\n"
            "• Reasoning → Groq Reasoning\n"
            "• Teknik/manufaktur → Gemini\n"
            "• Chat normal → Gemini\n"
            "• Foto/video → Gemini multimodal\n"
            "• Gambar → Pollinations jika aktif, lalu Gemini\n"
            "• Video → Pollinations jika aktif\n\n"
            "Perintah:\n"
            "/model\n"
            "/memory\n"
            "/knowledge\n"
            "/remember <teks>\n"
            "/forget <teks>\n"
            "/reset\n"
            "/gambar <prompt>\n"
            "/video <prompt>",
        )
        return

    # /model
    if is_command(text, "/model"):
        await send_text(
            chat_id,
            "🤖 STATUS PROVIDER\n\n"
            f"Gemini: {'✅' if gemini else '❌'} (dipilih: {GEMINI_MODEL}, aktif: {GEMINI_ACTIVE_MODEL})\n"
            f"Groq: {'✅' if groq else '❌'}\n"
            f"  Reasoning: {GROQ_REASONING_MODEL}\n"
            f"  Coding: {GROQ_CODING_MODEL}\n"
            f"  Fast: {GROQ_FAST_MODEL}\n"
            f"GitHub Memory: {'✅' if GITHUB_TOKEN else '❌'}\n"
            f"Pollinations: "
            f"{'✅' if POLLINATIONS_ENABLED and POLLINATIONS_KEY else '❌'}\n"
            f"  Image: {POLLINATIONS_IMAGE_MODEL}\n"
            f"  Video: {POLLINATIONS_VIDEO_MODEL}",
        )
        return

    # /remember
    if is_command(text, "/remember"):
        fact = command_arg(text)
        if not fact:
            await send_text(
                chat_id,
                "Gunakan: /remember <informasi yang ingin disimpan>",
            )
            return
        try:
            await remember_fact(uid, fact)
            await send_text(chat_id, "✅ Informasi disimpan ke memory GitHub.")
        except Exception as exc:
            log.exception("remember failed")
            await send_text(
                chat_id,
                f"❌ Gagal menyimpan memory: {str(exc)[:400]}",
            )
        return

    # /memory
    if is_command(text, "/memory"):
        try:
            mem = await load_memory(uid)
            facts = mem.get("facts", [])
            history = mem.get("history", [])[-10:]

            out = ["🧠 PERSISTENT MEMORY\n"]
            out.append(
                "FAKTA:\n"
                + (
                    "\n".join(f"• {x}" for x in facts)
                    if facts else "• Belum ada."
                )
            )
            out.append(
                "\nRIWAYAT TERAKHIR:\n"
                + (
                    "\n".join(
                        f"• {m.get('role')}: "
                        f"{clean_text(m.get('content'), 500)}"
                        for m in history
                    )
                    if history else "• Belum ada."
                )
            )
            await send_text(chat_id, "\n".join(out))
        except Exception as exc:
            log.exception("memory read failed")
            await send_text(
                chat_id,
                f"❌ Gagal membaca memory: {str(exc)[:400]}",
            )
        return

    # /forget
    if is_command(text, "/forget"):
        query = command_arg(text)
        if not query:
            await send_text(
                chat_id,
                "Gunakan: /forget <teks yang ingin dihapus>",
            )
            return
        try:
            removed = await forget_fact(uid, query)
            await send_text(
                chat_id,
                f"✅ Memory dihapus: {removed} item.",
            )
        except Exception as exc:
            log.exception("forget failed")
            await send_text(
                chat_id,
                f"❌ Gagal menghapus memory: {str(exc)[:400]}",
            )
        return

    # /knowledge
    if is_command(text, "/knowledge"):
        try:
            items = await github_list_dir(user_knowledge_dir(uid))
            count = sum(
                1 for x in items if x.get("type") == "file"
            )
            await send_text(
                chat_id,
                f"📚 Knowledge base: {count} chunk tersimpan di GitHub.\n\n"
                "Kirim file TXT, MD, CSV, JSON, PY, JS, TS, HTML, CSS, "
                "XML, YAML, YML, PDF, atau DOCX untuk dipelajari.",
            )
        except Exception as exc:
            log.exception("knowledge status failed")
            await send_text(
                chat_id,
                f"❌ Gagal membaca knowledge base: {str(exc)[:400]}",
            )
        return

    # /reset
    if is_command(text, "/reset"):
        try:
            await reset_user(uid)
            await send_text(
                chat_id,
                "🧹 Reset selesai.\n"
                "Memory, history, knowledge base, dan few-shot examples "
                "milik pengguna ini dihapus dari GitHub.",
            )
        except Exception as exc:
            log.exception("reset failed")
            await send_text(
                chat_id,
                f"❌ Reset gagal: {str(exc)[:400]}",
            )
        return

    # /gambar
    if is_command(text, "/gambar"):
        prompt = command_arg(text)
        if not prompt:
            await send_text(
                chat_id,
                "Contoh: /gambar desain pagar minimalis hitam modern",
            )
            return
        await send_text(chat_id, "🎨 Router gambar sedang bekerja...")
        try:
            data, provider = await generate_image(prompt)
            await send_photo(chat_id, data)
            await send_text(
                chat_id,
                f"✅ Gambar selesai. Provider: {provider}",
            )
        except Exception as exc:
            log.exception("image generation failed")
            await send_text(
                chat_id,
                f"❌ Generate gambar gagal: {str(exc)[:500]}",
            )
        return

    # /video
    if is_command(text, "/video"):
        prompt = command_arg(text)
        if not prompt:
            await send_text(
                chat_id,
                "Contoh: /video iklan pagar minimalis 10 detik",
            )
            return
        await send_text(chat_id, "🎬 Router video sedang bekerja...")
        try:
            data, provider = await generate_video(prompt)
            await send_video(chat_id, data)
            await send_text(
                chat_id,
                f"✅ Video selesai. Provider: {provider}",
            )
        except Exception as exc:
            log.exception("video generation failed")
            await send_text(
                chat_id,
                f"❌ Generate video gagal: {str(exc)[:500]}",
            )
        return

    # Telegram document / knowledge
    document = message.get("document")
    if document:
        filename = safe_name(document.get("file_name", "document"))
        ext = PurePosixPath(filename.lower()).suffix
        supported = {
            ".txt", ".md", ".csv", ".json", ".py", ".js", ".ts",
            ".html", ".css", ".xml", ".yaml", ".yml", ".pdf", ".docx",
        }

        if ext not in supported:
            await send_text(
                chat_id,
                "❌ Format file tidak didukung.",
            )
            return

        await send_text(
            chat_id,
            f"📚 Mempelajari {filename}...",
        )
        try:
            data, _ = await tg_file(document["file_id"])
            chunks = await ingest_knowledge(uid, filename, data)
            await send_text(
                chat_id,
                f"✅ Selesai. {filename} disimpan sebagai {chunks} "
                "chunk di knowledge base GitHub.",
            )
        except Exception as exc:
            log.exception("knowledge ingestion failed")
            await send_text(
                chat_id,
                f"❌ Gagal mempelajari file: {str(exc)[:500]}",
            )
        return

    # Photo analysis
    photo = message.get("photo")
    if photo:
        await send_text(chat_id, "🖼️ Sedang menganalisis foto...")
        try:
            data, path = await tg_file(photo[-1]["file_id"])
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            answer = await analyze_image(
                data,
                mime,
                caption or (
                    "Analisis foto ini secara detail. "
                    "Jika terkait manufaktur/fabrikasi, jelaskan kondisi, "
                    "kemungkinan masalah, ukuran/komponen yang dapat "
                    "diidentifikasi secara visual, dan saran praktis. "
                    "Jangan mengarang data yang tidak terlihat."
                ),
            )
            await send_text(chat_id, answer)
        except Exception as exc:
            log.exception("photo analysis failed")
            await send_text(
                chat_id,
                f"❌ Analisis foto gagal: {str(exc)[:500]}",
            )
        return

    # Video analysis
    video = message.get("video")
    if video:
        await send_text(chat_id, "🎥 Sedang menganalisis video...")
        try:
            data, path = await tg_file(video["file_id"])
            mime = mimetypes.guess_type(path)[0] or "video/mp4"
            answer = await analyze_video(
                data,
                mime,
                caption or (
                    "Analisis video ini secara detail. Jelaskan apa yang "
                    "terlihat, proses yang sedang dilakukan, masalah yang "
                    "terlihat, risiko, dan saran perbaikan praktis."
                ),
            )
            await send_text(chat_id, answer)
        except Exception as exc:
            log.exception("video analysis failed")
            await send_text(
                chat_id,
                f"❌ Analisis video gagal: {str(exc)[:500]}",
            )
        return

    # Normal chat
    if not text:
        return

    try:
        await tg(
            "sendChatAction",
            {"chat_id": chat_id, "action": "typing"},
        )

        answer, provider, category = await route_chat(uid, text)

        # Persistent conversation history.
        await add_history(uid, "user", text)
        await add_history(uid, "assistant", answer)

        # Save strong user/assistant exchanges as few-shot examples only for
        # normal successful interactions; retrieval decides whether to use it.
        if len(text) >= 12 and len(answer) >= 40:
            try:
                await save_example(uid, text, answer, category)
            except Exception:
                log.exception("few-shot save failed")

        await send_text(chat_id, answer)
        log.info(
            "Chat selesai user=%s category=%s provider=%s",
            uid,
            category,
            provider,
        )
    except Exception as exc:
        log.exception("chat failed")
        await send_text(
            chat_id,
            "❌ Semua provider AI untuk permintaan ini gagal.\n"
            f"Detail: {str(exc)[:500]}",
        )


# ---------------------------------------------------------------------------
# HTTP ENDPOINTS
#
# IMPORTANT FOR VERCEL:
# - api/index.py is the Python function.
# - The route may be exposed as /api or rewritten from /api/webhook.
# - Both /webhook and /api/webhook are defined so the application itself
#   accepts either path when the platform forwards it.
# ---------------------------------------------------------------------------

async def webhook_impl(
    request: Request,
    secret: Optional[str],
) -> dict[str, bool]:
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook secret",
        )

    try:
        update = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON",
        )

    # Telegram retries failed webhooks. We return quickly after processing.
    # The actual AI call is awaited so Telegram sees the result only after
    # the handler finishes.
    await handle_update(update)
    return {"ok": True}


@app.post("/")
async def webhook_root(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    return await webhook_impl(
        request,
        x_telegram_bot_api_secret_token,
    )


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "Designmanufaktur Super AI Agent",
        "telegram": bool(TELEGRAM_TOKEN),
        "github_memory": bool(GITHUB_TOKEN),
        "gemini": bool(gemini),
        "groq": bool(groq),
        "pollinations": bool(POLLINATIONS_ENABLED and POLLINATIONS_KEY),
    }


@app.post("/webhook")
async def webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    return await webhook_impl(
        request,
        x_telegram_bot_api_secret_token,
    )


@app.post("/api/webhook")
async def webhook_legacy(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    return await webhook_impl(
        request,
        x_telegram_bot_api_secret_token,
    )
