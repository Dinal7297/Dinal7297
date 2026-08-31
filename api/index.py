import os
import asyncio
import aiohttp

# ==========================================
# KONFIGURASI MODE /fast
# ==========================================
FAST_TIMEOUT = 1.5  # Batas maksimal respons provider (detik)

async def call_nvidia(prompt: str) -> dict | None:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FAST_TIMEOUT)) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    return {"provider": "NVIDIA", "text": text}
    except Exception:
        pass
    return None

async def call_openrouter(prompt: str) -> dict | None:
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: return None
    
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }
    payload = {
        "model": "cohere/north-mini-code:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FAST_TIMEOUT)) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    return {"provider": "OpenRouter FREE", "text": text}
    except Exception:
        pass
    return None

async def call_gemini(prompt: str) -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.3}
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=FAST_TIMEOUT)) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return {"provider": "Gemini FREE", "text": text}
    except Exception:
        pass
    return None

async def route_fast(prompt: str) -> str:
    """
    RACE MODE: Menembak ketiga API secara bersamaan.
    Provider yang merespons pertama kali dengan status 200 akan diambil hasilnya.
    Task yang lambat atau gagal akan langsung dibatalkan (cancel).
    """
    tasks = [
        asyncio.create_task(call_nvidia(prompt)),
        asyncio.create_task(call_openrouter(prompt)),
        asyncio.create_task(call_gemini(prompt))
    ]
    
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            # Batalkan request lain yang masih berjalan agar hemat resource server
            for t in tasks:
                if not t.done():
                    t.cancel()
            
            # Langsung kembalikan teks respons
            return result["text"]
            
    return "Maaf, semua provider AI sedang sibuk/timeout."

# ==========================================
# CONTOH INTEGRASI KE HANDLER TELEGRAM BOT
# ==========================================
# async def handle_fast_command(message):
#     user_prompt = message.text.replace('/fast ', '', 1).strip()
#     if not user_prompt:
#         await message.reply("Ketik pesan setelah /fast")
#         return
#     
#     # Panggil fungsi route_fast
#     fast_response = await route_fast(user_prompt)
#     await message.reply(fast_response)

