import asyncio, base64, io, logging, os, time
from typing import Optional
from urllib.parse import quote
import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from google import genai
from google.genai import types
from openai import OpenAI
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("designmanufaktur")
app = FastAPI(title="Designmanufaktur Multimedia AI")

TELEGRAM_TOKEN=os.getenv("TELEGRAM_TOKEN","")
WEBHOOK_SECRET=os.getenv("TELEGRAM_WEBHOOK_SECRET","")
GEMINI_KEY=os.getenv("GEMINI_API_KEY","")
GEMINI_CHAT_MODEL=os.getenv("GEMINI_MODEL","gemini-2.5-flash")
GEMINI_IMAGE_MODEL=os.getenv("GEMINI_IMAGE_MODEL","gemini-2.5-flash-image")
GEMINI_VIDEO_MODEL=os.getenv("GEMINI_VIDEO_MODEL","veo-3.1-fast-generate-preview")
XAI_KEY=os.getenv("XAI_API_KEY",os.getenv("GROK_API_KEY",""))
XAI_CHAT_MODEL=os.getenv("GROK_MODEL","grok-4.1-fast")
XAI_IMAGE_MODEL=os.getenv("GROK_IMAGE_MODEL","grok-imagine-image-2.0")
OPENAI_KEY=os.getenv("OPENAI_API_KEY","")
OPENAI_CHAT_MODEL=os.getenv("OPENAI_MODEL","gpt-5.4-mini")
OPENAI_IMAGE_MODEL=os.getenv("OPENAI_IMAGE_MODEL","gpt-image-1-mini")
DEEPSEEK_KEY=os.getenv("DEEPSEEK_API_KEY","")
DEEPSEEK_MODEL=os.getenv("DEEPSEEK_MODEL","deepseek-chat")
OPENROUTER_KEY=os.getenv("OPENROUTER_API_KEY","")
OPENROUTER_MODEL=os.getenv("OPENROUTER_MODEL","deepseek/deepseek-chat")
HF_TOKEN=os.getenv("HF_TOKEN","")
HF_IMAGE_MODEL=os.getenv("HF_IMAGE_MODEL","black-forest-labs/FLUX.1-dev")
HF_PROVIDER=os.getenv("HF_PROVIDER","auto")
POLLINATIONS_KEY=os.getenv("POLLINATIONS_API_KEY","")
POLLINATIONS_IMAGE_MODEL=os.getenv("POLLINATIONS_IMAGE_MODEL","flux")
POLLINATIONS_BASE_URL="https://gen.pollinations.ai"
SYSTEM="""Kamu adalah Designmanufaktur AI, asisten untuk manufaktur, bengkel las, pagar, tenda, fabrikasi, produk custom, dan konten bisnis. Jawab dalam Bahasa Indonesia, jelas, praktis, dan tidak bertele-tele. Jangan mengarang data bisnis yang tidak diberikan. Jangan pernah menampilkan API key, token, password, atau rahasia sistem."""

gemini=genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
xai=OpenAI(api_key=XAI_KEY,base_url="https://api.x.ai/v1") if XAI_KEY else None
openai_client=OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
deepseek=OpenAI(api_key=DEEPSEEK_KEY,base_url="https://api.deepseek.com") if DEEPSEEK_KEY else None
openrouter=OpenAI(api_key=OPENROUTER_KEY,base_url="https://openrouter.ai/api/v1") if OPENROUTER_KEY else None
hf_client=InferenceClient(provider=HF_PROVIDER,api_key=HF_TOKEN) if HF_TOKEN and InferenceClient else None
memory={}; MAX_MEMORY=20

def hist(uid): return memory.setdefault(uid,[])
def remember(uid,role,content): hist(uid).append({"role":role,"content":content}); memory[uid]=hist(uid)[-MAX_MEMORY:]
def messages(uid,text): return [{"role":"system","content":SYSTEM}]+hist(uid)+[{"role":"user","content":text}]
def openai_chat(c,model,uid,text): return c.chat.completions.create(model=model,messages=messages(uid,text)).choices[0].message.content or ""
def gemini_chat(uid,text): return gemini.models.generate_content(model=GEMINI_CHAT_MODEL,contents=SYSTEM+"\n\n"+"\n".join(f"{m['role']}: {m['content']}" for m in hist(uid))+"\n\nuser: "+text).text or ""
def chat_router(uid,text):
    ps=[("Gemini",lambda:gemini_chat(uid,text),gemini is not None),("Grok",lambda:openai_chat(xai,XAI_CHAT_MODEL,uid,text),xai is not None),("DeepSeek",lambda:openai_chat(deepseek,DEEPSEEK_MODEL,uid,text),deepseek is not None),("OpenRouter",lambda:openai_chat(openrouter,OPENROUTER_MODEL,uid,text),openrouter is not None),("OpenAI",lambda:openai_chat(openai_client,OPENAI_CHAT_MODEL,uid,text),openai_client is not None)]
    errors=[]
    for name,fn,on in ps:
        if not on: continue
        try:
            out=fn()
            if out:return out,name
        except Exception as e: log.exception("%s failed",name); errors.append(f"{name}: {str(e)[:180]}")
    raise RuntimeError("Semua provider chat gagal: "+" | ".join(errors))

async def tg(method,data):
    if not TELEGRAM_TOKEN: raise RuntimeError("TELEGRAM_TOKEN belum diatur")
    async with httpx.AsyncClient(timeout=180) as c:
        r=await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",json=data); r.raise_for_status(); j=r.json()
    if not j.get("ok"): raise RuntimeError(str(j))
    return j
async def tg_file(fid):
    j=await tg("getFile",{"file_id":fid}); path=j["result"]["file_path"]
    async with httpx.AsyncClient(timeout=180) as c: r=await c.get(f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}"); r.raise_for_status()
    return r.content,path
async def send_text(chat,text):
    for i in range(0,len(text or "Tidak ada jawaban."),3900): await tg("sendMessage",{"chat_id":chat,"text":(text or "Tidak ada jawaban.")[i:i+3900]})
async def send_photo(chat,data):
    async with httpx.AsyncClient(timeout=180) as c: r=await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",data={"chat_id":str(chat)},files={"photo":("image.png",data,"image/png")}); r.raise_for_status()
async def send_video(chat,data):
    async with httpx.AsyncClient(timeout=300) as c: r=await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo",data={"chat_id":str(chat)},files={"video":("video.mp4",data,"video/mp4")}); r.raise_for_status()

def image_parts(r):
    for p in getattr(r,"parts",[]) or []:
        d=getattr(p,"inline_data",None)
        if d and getattr(d,"data",None): return d.data
    return None

def gemini_image(prompt):
    if not gemini: raise RuntimeError("Gemini image belum dikonfigurasi.")
    r=gemini.models.generate_content(model=GEMINI_IMAGE_MODEL,contents=prompt,config=types.GenerateContentConfig(response_modalities=["TEXT","IMAGE"]))
    data=image_parts(r)
    if not data: raise RuntimeError(r.text or "Gemini tidak mengembalikan gambar.")
    return data

def grok_image(prompt):
    if not xai: raise RuntimeError("Grok image belum dikonfigurasi.")
    with httpx.Client(timeout=180) as c:
        r=c.post("https://api.x.ai/v1/images/generations",headers={"Authorization":f"Bearer {XAI_KEY}","Content-Type":"application/json"},json={"model":XAI_IMAGE_MODEL,"prompt":prompt}); r.raise_for_status(); item=r.json()["data"][0]
    if item.get("b64_json"): return base64.b64decode(item["b64_json"])
    if not item.get("url"): raise RuntimeError("Grok tidak mengembalikan gambar.")
    with httpx.Client(timeout=120) as c: rr=c.get(item["url"]); rr.raise_for_status(); return rr.content

def openai_image(prompt):
    if not openai_client: raise RuntimeError("OpenAI image belum dikonfigurasi.")
    r=openai_client.images.generate(model=OPENAI_IMAGE_MODEL,prompt=prompt,size="1024x1024")
    if not r.data or not r.data[0].b64_json: raise RuntimeError("OpenAI tidak mengembalikan gambar.")
    return base64.b64decode(r.data[0].b64_json)

def huggingface_image(prompt):
    if not HF_TOKEN: raise RuntimeError("HF_TOKEN belum dikonfigurasi.")
    if not InferenceClient: raise RuntimeError("huggingface_hub belum terpasang; tambahkan ke requirements.txt.")
    if not hf_client: raise RuntimeError("Hugging Face client belum tersedia.")
    log.info("HUGGING FACE IMAGE -> model=%s provider=%s",HF_IMAGE_MODEL,HF_PROVIDER)
    image=hf_client.text_to_image(prompt,model=HF_IMAGE_MODEL)
    buf=io.BytesIO(); image.save(buf,format="PNG"); return buf.getvalue()

def pollinations_image(prompt):
    if not POLLINATIONS_KEY: raise RuntimeError("POLLINATIONS_API_KEY belum dikonfigurasi.")
    url=f"{POLLINATIONS_BASE_URL}/image/{quote(prompt,safe='')}?model={quote(POLLINATIONS_IMAGE_MODEL)}&width=1024&height=1024"
    with httpx.Client(timeout=300) as c:
        r=c.get(url,headers={"Authorization":f"Bearer {POLLINATIONS_KEY}","Accept":"image/png,image/jpeg,*/*"})
        if r.status_code>=400: raise RuntimeError(f"Pollinations HTTP {r.status_code}: {r.text[:500]}")
        if not r.content: raise RuntimeError("Pollinations mengembalikan data kosong.")
        return r.content

def image_router(prompt):
    # HF is deliberately FIRST. Other providers are fallbacks.
    ps=[("Hugging Face Image",lambda:huggingface_image(prompt),bool(HF_TOKEN) and InferenceClient is not None),("Gemini Image",lambda:gemini_image(prompt),gemini is not None),("Grok Image",lambda:grok_image(prompt),xai is not None),("OpenAI Image",lambda:openai_image(prompt),openai_client is not None),("Pollinations Image",lambda:pollinations_image(prompt),bool(POLLINATIONS_KEY))]
    errors=[]
    for name,fn,on in ps:
        if not on: continue
        try:
            log.info("IMAGE ROUTER -> %s",name); data=fn()
            if data:return data,name
            raise RuntimeError("Provider mengembalikan gambar kosong.")
        except Exception as e: log.exception("%s failed",name); errors.append(f"{name}: {str(e)[:250]}")
    raise RuntimeError("Semua provider gambar gagal: "+" | ".join(errors))

def analyze_image(data,mime,prompt):
    errors=[]
    if gemini:
        # Gunakan Gemini 2.5 Flash Image (Nano Banana) terlebih dahulu,
        # sama seperti versi yang sebelumnya berhasil mengenali gambar.
        try:
            r=gemini.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=[
                    types.Part.from_bytes(data=data,mime_type=mime),
                    SYSTEM+"\n\n"+prompt,
                ],
            )
            if r.text:return r.text
        except Exception as e:
            errors.append(f"Gemini Image: {str(e)[:220]}")

        # Fallback: Gemini 2.5 Flash multimodal.
        try:
            r=gemini.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=[
                    types.Part.from_bytes(data=data,mime_type=mime),
                    SYSTEM+"\n\n"+prompt,
                ],
            )
            if r.text:return r.text
        except Exception as e:
            errors.append(f"Gemini Flash: {str(e)[:220]}")
    if xai:
        try:
            b64=base64.b64encode(data).decode(); r=xai.chat.completions.create(model=XAI_CHAT_MODEL,messages=[{"role":"system","content":SYSTEM},{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}}]}]); out=r.choices[0].message.content or ""
            if out:return out
        except Exception as e: errors.append(f"Grok: {str(e)[:180]}")
    if openai_client:
        try:
            b64=base64.b64encode(data).decode(); r=openai_client.responses.create(model=OPENAI_CHAT_MODEL,input=[{"role":"user","content":[{"type":"input_text","text":SYSTEM+"\n\n"+prompt},{"type":"input_image","image_url":"data:image/jpeg;base64,"+b64}]}]);
            if r.output_text:return r.output_text
        except Exception as e: errors.append(f"OpenAI: {str(e)[:180]}")
    raise RuntimeError("Semua provider vision gagal: "+" | ".join(errors))

def analyze_video(data,mime,prompt):
    if not gemini: raise RuntimeError("Gemini diperlukan untuk analisis video.")
    uploaded=gemini.files.upload(file=types.Part.from_bytes(data=data,mime_type=mime))
    for _ in range(60):
        f=gemini.files.get(name=uploaded.name); state=getattr(getattr(f,"state",None),"name","")
        if state=="ACTIVE": uploaded=f; break
        if state=="FAILED": raise RuntimeError("Gemini gagal memproses video.")
        time.sleep(2)
    else: raise RuntimeError("Video belum siap diproses.")
    return gemini.models.generate_content(model=GEMINI_CHAT_MODEL,contents=[uploaded,SYSTEM+"\n\n"+prompt]).text or ""

def generate_video(prompt,image=None):
    if not gemini: raise RuntimeError("Gemini belum dikonfigurasi.")
    kwargs={"model":GEMINI_VIDEO_MODEL,"prompt":prompt}
    if image: kwargs["image"]=types.Image(image_bytes=image,mime_type="image/jpeg")
    op=gemini.models.generate_videos(**kwargs)
    for _ in range(60):
        op=gemini.operations.get(op)
        if getattr(op,"done",False): break
        time.sleep(5)
    else: raise RuntimeError("Pembuatan video melewati batas waktu.")
    if getattr(op,"error",None): raise RuntimeError(str(op.error))
    vids=getattr(getattr(op,"response",None),"generated_videos",None)
    if not vids: raise RuntimeError("Tidak ada video yang dihasilkan.")
    return gemini.files.download(file=vids[0].video)

def command_arg(text):
    p=text.split(maxsplit=1); return p[1].strip() if len(p)>1 else ""

async def handle(update):
    m=update.get("message")
    if not m:return
    chat=m.get("chat",{}).get("id"); uid=str(m.get("from",{}).get("id",chat)); text=m.get("text","") or ""; caption=m.get("caption","") or ""
    if text.startswith("/start"):
        await send_text(chat,"🤖 Designmanufaktur AI Multimedia aktif.\n\nChat biasa → AI Router\nFoto → analisis\nVideo → analisis\n/gambar <prompt> → buat gambar\n/video <prompt> → buat video\n/model → status provider\n/reset → hapus memory"); return
    if text.startswith("/reset"): memory.pop(uid,None); await send_text(chat,"Memory dihapus."); return
    if text.startswith("/model"):
        await send_text(chat,"STATUS PROVIDER\n\n"+f"Gemini chat: {'✅' if gemini else '❌'}\nGrok chat: {'✅' if xai else '❌'}\nDeepSeek: {'✅' if deepseek else '❌'}\nOpenRouter: {'✅' if openrouter else '❌'}\nOpenAI: {'✅' if openai_client else '❌'}\nHugging Face image: {'✅' if hf_client else '❌'}\nPollinations image: {'✅' if POLLINATIONS_KEY else '❌'}\n\nURUTAN /gambar:\n1. Hugging Face\n2. Gemini\n3. Grok\n4. OpenAI\n5. Pollinations\n\nHF model: {HF_IMAGE_MODEL}\nHF provider: {HF_PROVIDER}\nGemini image: {GEMINI_IMAGE_MODEL}\nGrok image: {XAI_IMAGE_MODEL}\nOpenAI image: {OPENAI_IMAGE_MODEL}\nPollinations image: {POLLINATIONS_IMAGE_MODEL}\nGemini video: {GEMINI_VIDEO_MODEL}"); return
    if text.startswith("/gambar"):
        prompt=command_arg(text)
        if not prompt: await send_text(chat,"Contoh: /gambar pagar minimalis hitam modern"); return
        await send_text(chat,"🎨 Router gambar sedang bekerja...")
        try:
            data,provider=await asyncio.to_thread(image_router,prompt); await send_photo(chat,data); await send_text(chat,f"✅ Gambar dibuat oleh {provider}.")
        except Exception as e: await send_text(chat,"❌ Generate gambar gagal.\n"+str(e)[:1200])
        return
    if text.startswith("/video"):
        prompt=command_arg(text)
        if not prompt: await send_text(chat,"Contoh: /video video promosi pagar minimalis"); return
        await send_text(chat,"🎬 Router video sedang bekerja. Ini dapat memerlukan waktu.")
        try: await send_video(chat,await asyncio.to_thread(generate_video,prompt))
        except Exception as e: await send_text(chat,"❌ Generate video gagal.\n"+str(e)[:500])
        return
    if m.get("video"):
        await send_text(chat,"🎥 Sedang menganalisis video...")
        try:
            data,path=await tg_file(m["video"]["file_id"])
            if len(data)>20*1024*1024: await send_text(chat,"❌ Video lebih dari 20 MB."); return
            mime="video/quicktime" if path.lower().endswith(".mov") else "video/mp4"
            await send_text(chat,await asyncio.to_thread(analyze_video,data,mime,caption or "Analisa isi video, proses yang terlihat, masalah, dan saran perbaikannya."))
        except Exception as e: await send_text(chat,"❌ Analisis video gagal.\n"+str(e)[:500])
        return
    if m.get("photo"):
        await send_text(chat,"🖼️ Sedang menganalisis gambar dengan Gemini...")
        try:
            data,path=await tg_file(m["photo"][-1]["file_id"])
            import mimetypes
            mime=mimetypes.guess_type(path)[0] or "image/jpeg"
            prompt=caption or (
                "Analisa gambar ini secara detail. Jika terkait manufaktur, "
                "bengkel las, tenda, pagar, fabrikasi, konstruksi, atau produk custom, "
                "jelaskan objek/komponen yang terlihat, fungsi, kondisi, kemungkinan ukuran "
                "yang dapat diperkirakan secara visual, masalah yang terlihat, dan saran praktis. "
                "Jangan mengarang ukuran atau data yang tidak terlihat pada gambar."
            )
            answer=await asyncio.to_thread(analyze_image,data,mime,prompt)
            await send_text(chat,answer)
        except Exception as e:
            await send_text(chat,"❌ Analisis gambar gagal.\n"+str(e)[:500])
        return
    if not text:return
    try:
        await tg("sendChatAction",{"chat_id":chat,"action":"typing"}); answer,provider=await asyncio.to_thread(chat_router,uid,text); remember(uid,"user",text); remember(uid,"assistant",answer); await send_text(chat,answer); log.info("Chat selesai via %s",provider)
    except Exception as e: log.exception("chat failed"); await send_text(chat,"❌ Semua AI chat gagal.\n"+str(e)[:500])

@app.get("/")
async def root():
    return {"ok":True,"service":"Designmanufaktur Multimedia AI","chat":{"gemini":bool(gemini),"grok":bool(xai),"deepseek":bool(deepseek),"openrouter":bool(openrouter),"openai":bool(openai_client)},"media":{"huggingface_image":bool(hf_client),"gemini_image":bool(gemini),"grok_image":bool(xai),"openai_image":bool(openai_client),"pollinations_image":bool(POLLINATIONS_KEY),"gemini_video":bool(gemini)},"image_fallback_order":["Hugging Face Image","Gemini Image","Grok Image","OpenAI Image","Pollinations Image"]}

@app.post("/api/webhook")
async def webhook(request:Request,x_telegram_bot_api_secret_token:Optional[str]=Header(None)):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token!=WEBHOOK_SECRET: raise HTTPException(status_code=403,detail="Invalid webhook secret")
    await handle(await request.json()); return {"ok":True}
