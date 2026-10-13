"""Live NAIWA — LoRA edition, 3 input modes (SD1.5 + naiwa LoRA + LCM + TAESD).

Phone (HTTPS, same WiFi) streams camera over WebSocket -> PC GPU -> naiwa.
Latest frame written to data_raw/live/latest_lora.jpg for tools/live_view_lora.py.

Modes (switchable on the phone page):
  img       图片模式   raw frame -> img2img (fast ~8fps)
  edge_all  全场边缘   full-frame HED edge -> ControlNet (everything becomes naiwa)
  edge_cut  主体边缘   u2net cutout + HED -> ControlNet (only the subject becomes naiwa)

guide is fixed at 1.0 (no CFG -> single forward per step, 2x faster).
LoRA weight slider applies to ALL modes (shared UNet adapters).
Presets: stable / naiwa / chaos / melt.

Run:  python tools/live_web_lora.py      ->  https://<lan-ip>:7863
"""

import asyncio
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_naiwa as app  # noqa: E402

LIVE_DIR = os.path.join(app.HERE, "data_raw", "live")
os.makedirs(LIVE_DIR, exist_ok=True)
LATEST = os.path.join(LIVE_DIR, "latest_lora.jpg")
ORIG = os.path.join(LIVE_DIR, "latest_orig.jpg")

SEED = 42
GUIDE = 1.0
SIZE = int(os.environ.get("NAIWA_LIVE_SIZE", "512"))
CUT_EVERY = int(os.environ.get("NAIWA_LIVE_CUT_EVERY", "4"))

PRESETS = {
    "stable": {"label": "稳稳奶娃", "w": 1.2, "esteps": 4, "isteps": 4, "strength": 0.85},
    "naiwa": {"label": "标准奶娃", "w": 1.5, "esteps": 3, "isteps": 3, "strength": 0.8},
    "chaos": {"label": "抽象崩坏", "w": 2.4, "esteps": 3, "isteps": 3, "strength": 0.9},
    "melt": {"label": "糊化崩溃", "w": 1.8, "esteps": 5, "isteps": 5, "strength": 0.8},
}
MODES = {
    "img": "图片模式",
    "edge_all": "全场边缘",
    "edge_cut": "主体边缘",
}
DEFAULT_PRESET = os.environ.get("NAIWA_LIVE_PRESET", "naiwa")
DEFAULT_MODE = os.environ.get("NAIWA_LIVE_MODE", "img")

api = FastAPI()

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiveNaiwa</title>
<style>
 body{background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center;margin:0;padding:12px}
 canvas{max-width:92vw;max-height:52vh;border-radius:12px}
 #s{margin:10px;font-size:15px;color:#8f8}
 .row{margin:8px 0}
 button.p{margin:4px;padding:9px 14px;font-size:14px;border:0;border-radius:10px;background:#333;color:#ddd}
 button.p.on{background:#2f6f4f;color:#fff}
 button.m{margin:4px;padding:9px 14px;font-size:14px;border:0;border-radius:10px;background:#333;color:#ddd}
 button.m.on{background:#7a4fbf;color:#fff}
 #t{opacity:.6;font-size:12px;margin-top:8px}
 #flip{margin:4px;padding:9px 14px;font-size:14px;border:0;border-radius:10px;background:#444;color:#ddd}
 .cb{font-size:14px;margin:0 8px;display:inline-block}
</style></head><body>
<h3>LiveNaiwa 手机直播 → 电脑端显示</h3>
<div id="s">正在请求摄像头…</div>
<canvas id="c"></canvas>
<div class="row" id="modes"></div>
<div class="row" id="presets"></div>
<div class="row">LoRA 强度 <input type="range" id="lw" min="0.6" max="2.6" step="0.1" value="1.5" style="vertical-align:middle;width:170px"> <span id="lwv">1.5</span></div>
<div class="row">
  <label class="cb"><input type="checkbox" id="gray"> 黑白+锐化</label>
  <label class="cb"><input type="checkbox" id="cfg"> CFG×1.5（慢一倍）</label>
</div>
<div class="row"><button id="flip">切换前后摄像头</button></div>
<div id="t">图片模式最快；全场边缘=画面里一切都会变奶娃；主体边缘=只把你抠出来变一只奶娃。</div>
<video id="v" autoplay playsinline muted style="display:none"></video>
<script>
const v=document.getElementById('v'), c=document.getElementById('c'), s=document.getElementById('s'), flip=document.getElementById('flip');
let ws=null, last=0, facing='user', cur='naiwa', mode='img';
const MODES=[['img','图片模式'],['edge_all','全场边缘'],['edge_cut','主体边缘']];
const PRESETS=[['stable','稳稳奶娃',1.2],['naiwa','标准奶娃',1.5],['chaos','抽象崩坏',2.4],['melt','糊化崩溃',1.8]];
const mrow=document.getElementById('modes'), prow=document.getElementById('presets');
const lw=document.getElementById('lw'), lwv=document.getElementById('lwv');
MODES.forEach(([k,label])=>{
  const b=document.createElement('button');
  b.className='m'+(k===mode?' on':''); b.id='m_'+k; b.textContent=label;
  b.onclick=()=>{ mode=k; MODES.forEach(([k2])=>{document.getElementById('m_'+k2).className='m'+(k2===k?' on':'');}); if(ws&&ws.readyState===1) ws.send(JSON.stringify({mode:k})); };
  mrow.appendChild(b);
});
PRESETS.forEach(([k,label,w])=>{
  const b=document.createElement('button');
  b.className='p'+(k===cur?' on':''); b.id='p_'+k; b.textContent=label;
  b.onclick=()=>{ cur=k; lw.value=w; lwv.textContent=w; PRESETS.forEach(([k2])=>{document.getElementById('p_'+k2).className='p'+(k2===k?' on':'');}); if(ws&&ws.readyState===1) ws.send(JSON.stringify({preset:k})); };
  prow.appendChild(b);
});
lw.addEventListener('input',()=>{ lwv.textContent=lw.value; });
lw.addEventListener('change',()=>{ if(ws&&ws.readyState===1) ws.send(JSON.stringify({lora_w:parseFloat(lw.value)})); });
const grayCb=document.getElementById('gray'), cfgCb=document.getElementById('cfg');
grayCb.addEventListener('change',()=>{ if(ws&&ws.readyState===1) ws.send(JSON.stringify({gray:grayCb.checked})); });
cfgCb.addEventListener('change',()=>{ if(ws&&ws.readyState===1) ws.send(JSON.stringify({cfg:cfgCb.checked})); });
function loop(){
  requestAnimationFrame(loop);
  if(!ws || ws.readyState!==1 || v.videoWidth===0) return;
  const now=performance.now();
  if(now-last<120) return;  // ~8 fps send rate
  last=now;
  const maxw=640;
  const sc=Math.min(1, maxw/Math.max(v.videoWidth,v.videoHeight));
  c.width=Math.round(v.videoWidth*sc); c.height=Math.round(v.videoHeight*sc);
  c.getContext('2d').drawImage(v,0,0,c.width,c.height);
  c.toBlob(b=>{ if(b && ws.bufferedAmount<393216) ws.send(b); },'image/jpeg',0.72);
}
async function startCam(){
  const st=await navigator.mediaDevices.getUserMedia({video:{facingMode:facing},audio:false});
  if(v.srcObject) v.srcObject.getTracks().forEach(t=>t.stop());
  v.srcObject=st; await v.play();
}
function connectWS(){
  ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  ws.onmessage=e=>{ s.textContent='服务器 '+e.data; };
  ws.onclose=e=>{ s.textContent='连接断开 code='+e.code+' '+(e.reason||'')+' — 刷新重连'; };
  ws.onopen=()=>{ s.textContent='已连接'; ws.send(JSON.stringify({preset:cur})); ws.send(JSON.stringify({mode:mode})); ws.send(JSON.stringify({gray:grayCb.checked})); ws.send(JSON.stringify({cfg:cfgCb.checked})); };
}
flip.onclick=async()=>{
  facing = facing==='user' ? 'environment' : 'user';
  try{ await startCam(); }catch(e){ s.textContent='切换失败：'+e; }
};
async function start(){
  await startCam();
  s.textContent='摄像头就绪，连接服务器…';
  connectWS();
  loop();
}
start().catch(e=>{ s.textContent='摄像头失败：'+e; });
</script></body></html>"""

_edge_pipe = None
_img_pipe = None
_pe = None
_ne = None
_hed = None
_sess = None
_mask_state = {"mask": None, "age": 10 ** 9}


def get_session():
    global _sess
    if _sess is None:
        from rembg import new_session

        _sess = new_session("u2net")
    return _sess


def get_pipe():
    global _edge_pipe, _img_pipe, _pe, _ne, _hed
    if _edge_pipe is not None:
        return _edge_pipe, _img_pipe, _pe, _ne
    from diffusers import (
        AutoencoderTiny,
        ControlNetModel,
        LCMScheduler,
        StableDiffusionControlNetPipeline,
        StableDiffusionImg2ImgPipeline,
    )

    t0 = time.time()
    cn = ControlNetModel.from_pretrained(app.CN_DIR, torch_dtype=torch.float16)
    ep = StableDiffusionControlNetPipeline.from_pretrained(
        app.SD15_BASE, controlnet=cn, torch_dtype=torch.float16, variant="fp16", safety_checker=None
    )
    ep.load_lora_weights(app.LORA_DIR, adapter_name="naiwa")
    ep.load_lora_weights(app.LCM_DIR, adapter_name="lcm")
    pres = PRESETS[DEFAULT_PRESET]
    ep.set_adapters(["naiwa", "lcm"], adapter_weights=[pres["w"], 1.0])
    sched_cfg = ep.scheduler.config
    ep.scheduler = LCMScheduler.from_config(sched_cfg)
    taesd_dir = os.path.join(app.HERE, "models", "taesd")
    if os.path.exists(os.path.join(taesd_dir, "diffusion_pytorch_model.safetensors")):
        try:
            ep.vae = AutoencoderTiny.from_pretrained(taesd_dir, torch_dtype=torch.float16)
            print("[pipe] TAESD tiny VAE enabled", flush=True)
        except Exception as e:
            print("[pipe] TAESD failed, using full VAE:", repr(e), flush=True)
    ep.to(app.DEVICE)
    ep.set_progress_bar_config(disable=True)
    ip = StableDiffusionImg2ImgPipeline(
        vae=ep.vae,
        text_encoder=ep.text_encoder,
        tokenizer=ep.tokenizer,
        unet=ep.unet,
        scheduler=LCMScheduler.from_config(sched_cfg),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    ip.set_progress_bar_config(disable=True)
    with torch.no_grad():
        _pe, _ne = ep.encode_prompt(
            prompt=app.LORA_PROMPT, negative_prompt=app.LORA_NEG, device=app.DEVICE,
            num_images_per_prompt=1, do_classifier_free_guidance=True,
        )
    _hed = app.get_hed()
    _edge_pipe, _img_pipe = ep, ip
    print(f"[pipe] loaded in {time.time() - t0:.1f}s (img+edge share one UNet)", flush=True)
    return _edge_pipe, _img_pipe, _pe, _ne


def _fit(bgr, max_side=SIZE):
    """Full frame, aspect-ratio preserved, longest side = max_side (multiple of 8)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    sc = max_side / max(w, h)
    nw = max(8, int(round(w * sc / 8)) * 8)
    nh = max(8, int(round(h * sc / 8)) * 8)
    return np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BICUBIC))


def _get_mask(rgb):
    _mask_state["age"] += 1
    if _mask_state["mask"] is None or _mask_state["age"] >= CUT_EVERY:
        _mask_state["age"] = 0
        from rembg import remove

        cut = remove(Image.fromarray(rgb), session=get_session()).convert("RGBA")
        a = np.asarray(cut)[..., 3]
        m = (a > 128).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        m = cv2.GaussianBlur(m, (3, 3), 0)
        _mask_state["mask"] = m
    return _mask_state["mask"]


def _write_jpg(img, path, quality=92):
    tmp = path + ".tmp.jpg"
    img.save(tmp, quality=quality)
    for _ in range(15):
        try:
            os.replace(tmp, path)
            break
        except OSError:
            time.sleep(0.02)


def _gray_sharp(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 2.0)
    s = np.clip(g + 1.6 * (g - blur), 0, 255).astype(np.uint8)
    return np.stack([s, s, s], -1)


def work(data, ep, ip, pe, ne, preset, mode, gray, cfg):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return
    rgb = _fit(arr)
    if gray:
        rgb = _gray_sharp(rgb)
    _write_jpg(Image.fromarray(rgb), ORIG)
    gen = torch.Generator(device=app.DEVICE).manual_seed(SEED)
    if cfg:
        kw = dict(prompt_embeds=pe, negative_prompt_embeds=ne)
        gscale = 1.5
    else:
        kw = dict(prompt_embeds=pe)
        gscale = GUIDE
    if mode == "img":
        img = ip(
            image=Image.fromarray(rgb),
            strength=preset["strength"], num_inference_steps=preset["isteps"],
            guidance_scale=gscale, generator=gen, **kw,
        ).images[0]
    else:
        if mode == "edge_cut":
            mask = _get_mask(rgb)
            a = mask.astype(np.float32)[..., None] / 255.0
            src = (rgb.astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
        else:
            src = rgb
        edge = app.edge_postprocess(_hed(src))
        control = 255 - edge
        img = ep(
            image=Image.fromarray(np.stack([control] * 3, -1)),
            num_inference_steps=preset["esteps"], guidance_scale=gscale, generator=gen, **kw,
        ).images[0]
    _write_jpg(img, LATEST)


@api.get("/")
def index():
    return HTMLResponse(PAGE)


@api.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    print(f"[ws] client connected: {ws.client}", flush=True)
    ep, ip, pe, ne = await asyncio.to_thread(get_pipe)
    state = {
        "data": None,
        "done": False,
        "preset": dict(PRESETS[DEFAULT_PRESET]),
        "mode": DEFAULT_MODE,
        "gray": False,
        "cfg": False,
    }
    state["preset"]["name"] = DEFAULT_PRESET

    async def receiver():
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    state["data"] = msg["bytes"]  # latest-wins
                elif msg.get("text"):
                    try:
                        cmd = json.loads(msg["text"])
                        k = cmd.get("preset")
                        if k in PRESETS:
                            state["preset"] = dict(PRESETS[k])
                            state["preset"]["name"] = k
                            ep.set_adapters(["naiwa", "lcm"], adapter_weights=[PRESETS[k]["w"], 1.0])
                            await ws.send_text(f"已切换：{PRESETS[k]['label']}")
                        m = cmd.get("mode")
                        if m in MODES:
                            state["mode"] = m
                            await ws.send_text(f"模式：{MODES[m]}")
                        w = cmd.get("lora_w")
                        if isinstance(w, (int, float)):
                            state["preset"]["w"] = float(w)
                            state["preset"]["name"] = "custom"
                            state["preset"]["label"] = f"手动 w={w:.1f}"
                            ep.set_adapters(["naiwa", "lcm"], adapter_weights=[float(w), 1.0])
                            await ws.send_text(f"LoRA 强度 = {w:.1f}")
                        g = cmd.get("gray")
                        if isinstance(g, bool):
                            state["gray"] = g
                            await ws.send_text("黑白+锐化 " + ("开" if g else "关"))
                        c = cmd.get("cfg")
                        if isinstance(c, bool):
                            state["cfg"] = c
                            await ws.send_text("CFG×1.5 " + ("开" if c else "关"))
                    except Exception:
                        pass
        except Exception:
            pass
        state["done"] = True

    rt = asyncio.create_task(receiver())
    cnt = 0
    t0 = time.time()
    try:
        while True:
            data = state["data"]
            if data is None:
                if state["done"]:
                    break
                await asyncio.sleep(0.008)
                continue
            state["data"] = None
            preset = dict(state["preset"])
            mode = state["mode"]
            t1 = time.time()
            try:
                await asyncio.to_thread(work, data, ep, ip, pe, ne, preset, mode, state["gray"], state["cfg"])
            except Exception as e:
                print("[ws] frame error (kept alive):", repr(e), flush=True)
            cnt += 1
            dt_ms = (time.time() - t1) * 1000
            fps = cnt / max(time.time() - t0, 1e-6)
            flags = ("+灰锐 " if state["gray"] else "") + ("+CFG " if state["cfg"] else "")
            try:
                await ws.send_text(
                    f"[{MODES.get(mode, mode)}·{preset.get('label', preset['name'])} {flags}] #{cnt} {dt_ms:.0f} ms | {fps:.1f} fps"
                )
            except Exception:
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rt.cancel()
        print(f"[ws] client disconnected after {cnt} frames", flush=True)


def detect_lan_ip():
    import socket

    env = os.environ.get("NAIWA_LAN_IP")
    if env:
        return env
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def ensure_cert():
    certdir = os.path.join(app.HERE, "certs")
    os.makedirs(certdir, exist_ok=True)
    cert = os.path.join(certdir, "live_cert.pem")
    key = os.path.join(certdir, "live_key.pem")
    lan_ip = detect_lan_ip()
    if os.path.exists(cert) and os.path.exists(key):
        try:
            from cryptography import x509

            with open(cert, "rb") as f:
                existing = x509.load_pem_x509_certificate(f.read())
            san = existing.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            if any(str(ip) == lan_ip for ip in san.get_values_for_type(x509.IPAddress)):
                return cert, key
            print(f"LAN IP changed to {lan_ip}, regenerating cert ...", flush=True)
        except Exception:
            pass
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    ips = ["127.0.0.1", lan_ip]
    key_obj = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "naiwa-live")])
    san = x509.SubjectAlternativeName(
        [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips] + [x509.DNSName("localhost")]
    )
    now = datetime.datetime.utcnow()
    cert_obj = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key_obj.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(san, critical=False)
        .sign(key_obj, hashes.SHA256())
    )
    with open(key, "wb") as f:
        f.write(key_obj.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(cert, "wb") as f:
        f.write(cert_obj.public_bytes(serialization.Encoding.PEM))
    print("self-signed cert generated:", cert, flush=True)
    return cert, key


if __name__ == "__main__":
    import uvicorn

    print("preloading pipeline ...", flush=True)
    get_pipe()
    get_session()
    cert, key = ensure_cert()
    port = int(os.environ.get("NAIWA_LIVE_PORT", "7863"))
    lan = detect_lan_ip()
    print(f"phone page: https://{lan}:{port}   (证书警告 -> 继续访问)", flush=True)
    uvicorn.run(api, host="0.0.0.0", port=port, ssl_certfile=cert, ssl_keyfile=key, log_level="warning")
