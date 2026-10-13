"""Core library for the Naiwa apps (shared helpers, models, generation).

UIs live in:
  - app_pix2.py  (原版 pix2pix 256px, port 7861)
  - app_lora.py  (SD1.5 + naiwa LoRA + LCM + ControlNet 512px, port 7864)

Tools import this module for helpers: HED, rembg cutout, pix2pix generator, LoRA pipeline.
"""

import glob
import os
import secrets
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("REMBG_HOME", os.path.join(HERE, "models", "rembg"))
REPO = os.path.join(HERE, "pytorch-CycleGAN-and-pix2pix")
sys.path.insert(0, REPO)
from models import networks  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.environ.get("NAIWA_PIX2_DIR") or os.path.join(REPO, "checkpoints", "naiwa_pix2pix2")
SHOES_PTH = os.path.join(HERE, "pretrained", "edges2shoes.pth")
TEST_DIR = os.path.join(REPO, "datasets", "naiwa_final", "test")

EXPERIMENT = "naiwa_pix2pix2"
TOTAL_EPOCHS = 240

LORA_DIR = os.environ.get("NAIWA_LORA_DIR") or os.path.join(HERE, "models", "naiwa_lora", "out")
SD15_BASE = os.path.join(HERE, "models", "sd15-base")
CN_DIR = os.path.join(HERE, "models", "controlnet-scribble")
LCM_DIR = os.path.join(HERE, "models", "lcm-lora-sd15")
LORA_PROMPT = ("naiwa, a cute yellow round creature with a cream belly, "
               "big round eyes and a smiling mouth, on white background")
LORA_NEG = ("photo, realistic, 3d, blurry, low quality, dark background, colored background, "
            "gradient background, yellow background, orange background, colorful background, "
            "stripes, pattern, multiple creatures, faceless, eyeless")

_cache = {}
_hed = None
_lora_pipes = {}
_last_pick = None


def fresh_seed():
    return secrets.randbits(31)


def epoch_list():
    out = []
    for p in glob.glob(os.path.join(CKPT_DIR, "*_net_G.pth")):
        base = os.path.basename(p).split("_")[0]
        if base.isdigit():
            out.append(int(base))
    return sorted(out)


def model_choices():
    choices = []
    if os.path.exists(SHOES_PTH):
        choices.append("edges2shoes 预训练（鞋，看概念效果）")
    for e in epoch_list():
        choices.append(f"奶娃 epoch {e}")
    if os.path.exists(os.path.join(CKPT_DIR, "latest_net_G.pth")):
        choices.append("奶娃 latest（最新权重）")
    return choices


def default_choice():
    choices = model_choices()
    for c in choices:
        if "latest" in c:
            return c
    for c in choices:
        if "奶娃" in c:
            return c
    return choices[0] if choices else None


def ckpt_path(choice):
    if choice.startswith("edges2shoes"):
        return SHOES_PTH
    if "latest" in choice:
        return os.path.join(CKPT_DIR, "latest_net_G.pth")
    e = choice.split()[-1]
    return os.path.join(CKPT_DIR, f"{e}_net_G.pth")


def load_generator(choice):
    path = ckpt_path(choice)
    mtime = os.path.getmtime(path)
    cached = _cache.get(choice)
    if cached and cached[0] == mtime:
        return cached[1]
    net = networks.define_G(3, 3, 64, "unet_256", "batch", False, "normal", 0.02)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    net.to(DEVICE).eval()
    _cache[choice] = (mtime, net)
    return net


def get_hed():
    global _hed
    if _hed is None:
        sys.path.insert(0, os.path.join(HERE, "tools"))
        from hed import HEDDetector

        _hed = HEDDetector(device=DEVICE)
    return _hed


_rembg_sessions = {}


def get_subject_cutout(rgb, model="u2net"):
    """rembg AI 抠出主体并贴白底（alpha 硬阈值去软边）；失败时回退原图，绝不中断流程。"""
    try:
        from rembg import new_session, remove

        sess = _rembg_sessions.get(model)
        if sess is None:
            sess = new_session(model)
            _rembg_sessions[model] = sess
        cut = remove(Image.fromarray(rgb), session=sess).convert("RGBA")
        arr = np.asarray(cut)
        alpha = arr[..., 3]
        mask = (alpha > 128).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        a = mask.astype(np.float32)[..., None] / 255.0
        comp = arr[..., :3].astype(np.float32) * a + 255.0 * (1 - a)
        return comp.astype(np.uint8)
    except Exception:
        return rgb


def run_generator(net, gray256):
    x = gray256.astype(np.float32) / 255.0 * 2 - 1
    t = torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1).contiguous().to(DEVICE)
    with torch.no_grad():
        y = net(t)
    y = y[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(((y + 1) / 2 * 255).clip(0, 255).astype(np.uint8))


def gacha_transform(gray, rng):
    """Random small warp + stroke jitter so repeated generations differ (lottery mode)."""
    h, w = gray.shape
    angle = rng.uniform(-7, 7)
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-0.04, 0.04) * w
    ty = rng.uniform(-0.04, 0.04) * h
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    grid = 4
    dx = cv2.resize(rng.uniform(-5, 5, (grid + 1, grid + 1)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(rng.uniform(-5, 5, (grid + 1, grid + 1)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(out, xx + dx, yy + dy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    k = np.ones((3, 3), np.uint8)
    r = rng.random()
    if r < 0.4:
        out = cv2.dilate(out, k, iterations=1)
    elif r < 0.6:
        out = cv2.erode(out, k, iterations=1)
    return out


def extract_array(editor_value):
    if editor_value is None:
        return None
    v = editor_value
    if isinstance(v, dict):
        src = v.get("composite")
        if src is None:
            src = v.get("background")
        v = src
    if isinstance(v, str):
        try:
            v = np.array(Image.open(v).convert("RGB"))
        except Exception:
            return None
    return v


def to_gray256(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        a = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = arr[..., :3].astype(np.float32)
        arr = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
    if arr.ndim == 3:
        img = Image.fromarray(arr[..., :3]).convert("L")
    else:
        img = Image.fromarray(arr).convert("L")
    img = img.resize((256, 256), Image.BICUBIC)
    return np.asarray(img)


def to_rgb_square256(arr, size=256):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.shape[2] == 4:
        a = arr[..., 3:4].astype(np.float32) / 255.0
        arr = (arr[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    else:
        arr = arr[..., :3].astype(np.uint8)
    h, w = arr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    arr = arr[y0:y0 + s, x0:x0 + s]
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BICUBIC))


def edge_postprocess(edge):
    e = edge.astype(np.float32)
    e = np.clip((e - 50) * (255.0 / (200 - 50)), 0, 255)
    binimg = (e > 100).astype(np.uint8) * 255
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(binimg, (3, 3), 0)


def extract_ink(arr):
    """Auto-detect background color from border pixels; return ink mask (white strokes on black)."""
    a = np.asarray(arr)
    if a.ndim != 3:
        return None
    alpha = None
    if a.shape[2] == 4:
        alpha = a[..., 3].astype(np.float32) / 255.0
    rgb = a[..., :3].astype(np.float32)
    border = np.concatenate([
        rgb[:6, :].reshape(-1, 3), rgb[-6:, :].reshape(-1, 3),
        rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    diff = np.abs(rgb - bg).mean(axis=2)
    mask = np.clip(diff * (255.0 / 90.0), 0, 255)
    if alpha is not None and alpha.min() < 0.95:
        mask = np.maximum(mask, np.clip(alpha * 255.0, 0, 255))
    return mask.astype(np.uint8)


def new_canvas(hex_color):
    try:
        s = str(hex_color).lstrip("#")
        r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except Exception:
        r = g = b = 0
    img = np.zeros((512, 512, 3), np.uint8)
    img[:, :] = (r, g, b)
    return img


def find_choice(sub, prefix=None):
    for c in model_choices():
        if sub in c and (prefix is None or c.startswith(prefix)):
            return c
    return None


def random_outline():
    global _last_pick
    files = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    if not files:
        return None
    pick = np.random.choice(files)
    for _ in range(5):
        if len(files) == 1 or pick != _last_pick:
            break
        pick = np.random.choice(files)
    _last_pick = pick
    pair = np.array(Image.open(pick).convert("RGB"))
    return pair[:, : pair.shape[1] // 2]


def shoe_outline():
    p = os.path.join(HERE, "assets", "shoe_example.png")
    return np.array(Image.open(p).convert("RGB"))


def get_lora_pipe():
    if "cn" in _lora_pipes:
        return _lora_pipes["cn"]
    from diffusers import ControlNetModel, LCMScheduler, StableDiffusionControlNetPipeline

    t0 = time.time()
    cn = ControlNetModel.from_pretrained(CN_DIR, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD15_BASE, controlnet=cn, torch_dtype=torch.float16, variant="fp16", safety_checker=None
    )
    pipe.load_lora_weights(LORA_DIR, adapter_name="naiwa")
    pipe.load_lora_weights(LCM_DIR, adapter_name="lcm")
    pipe.set_adapters(["naiwa", "lcm"], adapter_weights=[1.4, 1.0])
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    _lora_pipes["cn"] = pipe
    print(f"[lora] pipeline loaded in {time.time() - t0:.1f}s", flush=True)
    return pipe


def pix2_generate(editor_value, choice, gacha=True):
    arr = extract_array(editor_value)
    mask = extract_ink(arr) if arr is not None else None
    if mask is None or int((mask > 64).sum()) < 40:
        return None, "先在画布上画一个轮廓"
    if not choice:
        return None, "没有可用模型"
    t0 = time.time()
    net = load_generator(choice)
    gray = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    if gacha:
        gray = gacha_transform(gray, np.random.default_rng())
    out = run_generator(net, gray)
    dt = (time.time() - t0) * 1000
    tag = "抽奖模式（每次随机微变）" if gacha else "固定输出"
    return out, f"模型: {choice} | {tag} | {dt:.0f} ms"


generate = pix2_generate


def pix2_generate_image(image, choice, gacha=True):
    if image is None:
        return None, None, None, "先上传一张图片"
    if not choice:
        return None, None, None, "没有可用模型"
    net = load_generator(choice)
    rgb = to_rgb_square256(image)
    t0 = time.time()
    cut = get_subject_cutout(rgb, "u2net")
    edge = edge_postprocess(get_hed()(cut))
    if gacha:
        edge = gacha_transform(edge, np.random.default_rng())
    out = run_generator(net, edge)
    dt = (time.time() - t0) * 1000
    edge_vis = Image.fromarray(np.stack([edge] * 3, -1))
    tag = "抽奖模式" if gacha else "固定输出"
    return cut, edge_vis, out, f"模型: {choice} | {tag} | 抠图+边缘+生成 {dt:.0f} ms"


def pix2_generate_video(video, choice, max_seconds=15.0, max_frames=300, sample_fps=12.0):
    path = extract_video_path(video)
    if not path:
        return None, None, "先上传一个视频"
    if not choice:
        return None, None, "没有可用模型"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, "无法读取视频文件"
    net = load_generator(choice)
    hed = get_hed()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    stride = max(1, int(round(fps / sample_fps)))
    out_fps = fps / stride
    out_dir = os.path.join(HERE, "data_raw", "video_out")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"naiwa_{ts}.mp4")
    cut_path = os.path.join(out_dir, f"cut_{ts}.mp4")
    vw = vwc = None
    n = 0
    idx = -1
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride != 0:
            continue
        if n >= max_frames or (idx / fps) > max_seconds:
            break
        rgb = to_rgb_square256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        rgb = get_subject_cutout(rgb, "u2net")
        edge = edge_postprocess(hed(rgb))
        out = run_generator(net, edge)
        bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, out_fps, (256, 256))
            vwc = cv2.VideoWriter(cut_path, fourcc, out_fps, (256, 256))
            if not vw.isOpened():
                out_path = out_path.replace(".mp4", ".avi")
                cut_path = cut_path.replace(".mp4", ".avi")
                fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                vw = cv2.VideoWriter(out_path, fourcc2, out_fps, (256, 256))
                vwc = cv2.VideoWriter(cut_path, fourcc2, out_fps, (256, 256))
        if vw.isOpened():
            vw.write(bgr)
            if vwc.isOpened():
                vwc.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            n += 1
    cap.release()
    if vw:
        vw.release()
    if vwc:
        vwc.release()
    dt = time.time() - t0
    if n == 0:
        return None, None, "没有处理到任何帧（视频太短或编码不支持）"
    return cut_path, out_path, f"模型: {choice} | {n} 帧 @ {out_fps:.1f} fps | 耗时 {dt:.1f}s"


def _pick_seed(true_random, seed):
    return fresh_seed() if true_random else int(seed)


def extract_video_path(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("video", "path", "name"):
            if isinstance(v.get(k), str):
                return v[k]
    return None


def lora_generate_sketch(editor_value, steps, lora_w, guidance, seed, true_random):
    arr = extract_array(editor_value)
    mask = extract_ink(arr) if arr is not None else None
    if mask is None or int((mask > 64).sum()) < 40:
        return None, "先在画布上画一个轮廓"
    if not os.path.exists(os.path.join(LORA_DIR, "pytorch_lora_weights.safetensors")):
        return None, "找不到 LoRA 权重（models/naiwa_lora/out）"
    try:
        t0 = time.time()
        pipe = get_lora_pipe()
        pipe.set_adapters(["naiwa", "lcm"], adapter_weights=[float(lora_w), 1.0])
        g = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LINEAR)
        control = (255 - g).astype(np.uint8)
        s = _pick_seed(true_random, seed)
        gen = torch.Generator(device=DEVICE).manual_seed(s)
        img = pipe(
            LORA_PROMPT, negative_prompt=LORA_NEG,
            image=Image.fromarray(np.stack([control] * 3, -1)),
            num_inference_steps=int(steps), guidance_scale=float(guidance), generator=gen,
        ).images[0]
        dt = (time.time() - t0) * 1000
        return img, f"LoRA ControlNet 512px | 强度 {float(lora_w):.1f} | {int(steps)} 步 | 引导 {float(guidance):.1f} | 种子 {s} | {dt:.0f} ms"
    except Exception as e:
        return None, f"生成失败: {e!r}"


def _pad_control(edge, scale=0.7):
    """Shrink white-on-black edge to `scale` and center on black canvas (gives white margin)."""
    h, w = edge.shape
    s = int(h * scale)
    small = cv2.resize(edge, (s, s), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w), np.uint8)
    o = (h - s) // 2
    canvas[o:o + s, o:o + s] = small
    return canvas


def _scale_mask(mask, scale=0.7):
    h, w = mask.shape
    s = int(h * scale)
    small = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((h, w), np.uint8)
    o = (h - s) // 2
    canvas[o:o + s, o:o + s] = small
    return canvas


def _white_composite(out_img, mask, erode=3, feather=3):
    """Keep output inside mask, white outside (soft edge)."""
    arr = np.asarray(out_img).astype(np.float32)
    m = mask.copy()
    if erode > 0:
        m = cv2.erode(m, np.ones((erode, erode), np.uint8))
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    a = m[..., None].astype(np.float32) / 255.0
    return Image.fromarray((arr * a + 255.0 * (1 - a)).astype(np.uint8))


def lora_generate_image(image, steps, lora_w, guidance, seed, true_random):
    if image is None:
        return None, None, None, "先上传一张图片"
    if not os.path.exists(os.path.join(LORA_DIR, "pytorch_lora_weights.safetensors")):
        return None, None, None, "找不到 LoRA 权重（models/naiwa_lora/out）"
    try:
        t0 = time.time()
        pipe = get_lora_pipe()
        pipe.set_adapters(["naiwa", "lcm"], adapter_weights=[float(lora_w), 1.0])
        rgb = to_rgb_square256(image, 512)
        cut = get_subject_cutout(rgb, "u2net")
        subj = ((cut.min(axis=2) < 245).astype(np.uint8) * 255)
        edge = edge_postprocess(get_hed()(cut))
        edge_pad = _pad_control(edge, 0.7)
        control = (255 - edge_pad).astype(np.uint8)
        s = _pick_seed(true_random, seed)
        gen = torch.Generator(device=DEVICE).manual_seed(s)
        out = pipe(
            LORA_PROMPT, negative_prompt=LORA_NEG,
            image=Image.fromarray(np.stack([control] * 3, -1)),
            num_inference_steps=int(steps), guidance_scale=float(guidance), generator=gen,
        ).images[0]
        out = _white_composite(out, _scale_mask(subj, 0.7))
        dt = (time.time() - t0) * 1000
        edge_vis = Image.fromarray(np.stack([edge_pad] * 3, -1))
        return cut, edge_vis, out, f"抠主体+HED边缘 → ControlNet（居中留白+白底合成） | 强度 {float(lora_w):.1f} | {int(steps)} 步 | 种子 {s} | {dt:.0f} ms"
    except Exception as e:
        return None, None, None, f"生成失败: {e!r}"


def lora_generate_video(video, steps, lora_w, guidance, seed, true_random,
                        max_seconds=8.0, max_frames=96, sample_fps=12.0):
    path = extract_video_path(video)
    if not path:
        return None, None, "先上传一个视频"
    if not os.path.exists(os.path.join(LORA_DIR, "pytorch_lora_weights.safetensors")):
        return None, None, "找不到 LoRA 权重（models/naiwa_lora/out）"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, "无法读取视频文件"
    pipe = get_lora_pipe()
    pipe.set_adapters(["naiwa", "lcm"], adapter_weights=[float(lora_w), 1.0])
    hed = get_hed()
    s = _pick_seed(true_random, seed)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    stride = max(1, int(round(fps / sample_fps)))
    out_fps = fps / stride
    out_dir = os.path.join(HERE, "data_raw", "video_out")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"naiwa_lora_{ts}.mp4")
    cut_path = os.path.join(out_dir, f"cut_lora_{ts}.mp4")
    vw = vwc = None
    n = 0
    idx = -1
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride != 0:
            continue
        if n >= max_frames or (idx / fps) > max_seconds:
            break
        rgb = to_rgb_square256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 512)
        rgb = get_subject_cutout(rgb, "u2net")
        edge = edge_postprocess(hed(rgb))
        control = (255 - edge).astype(np.uint8)
        gen = torch.Generator(device=DEVICE).manual_seed(s)
        out = pipe(
            LORA_PROMPT, negative_prompt=LORA_NEG,
            image=Image.fromarray(np.stack([control] * 3, -1)),
            num_inference_steps=int(steps), guidance_scale=float(guidance), generator=gen,
        ).images[0]
        bgr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, out_fps, (512, 512))
            vwc = cv2.VideoWriter(cut_path, fourcc, out_fps, (512, 512))
            if not vw.isOpened():
                out_path = out_path.replace(".mp4", ".avi")
                cut_path = cut_path.replace(".mp4", ".avi")
                fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                vw = cv2.VideoWriter(out_path, fourcc2, out_fps, (512, 512))
                vwc = cv2.VideoWriter(cut_path, fourcc2, out_fps, (512, 512))
        if vw.isOpened():
            vw.write(bgr)
            if vwc.isOpened():
                vwc.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            n += 1
    cap.release()
    if vw:
        vw.release()
    if vwc:
        vwc.release()
    dt = time.time() - t0
    if n == 0:
        return None, None, "没有处理到任何帧（视频太短或编码不支持）"
    return cut_path, out_path, f"LoRA 视频 | {n} 帧 @ {out_fps:.1f} fps | 种子 {s} | 耗时 {dt:.1f}s"


SWEEP_PARAMS = ["LoRA 强度", "步数", "引导", "种子"]


def _sweep_parse(s, param):
    vals = []
    for tok in str(s).replace("，", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            return None
        vals.append(max(1, int(v)) if param == "步数" else int(v) if param == "种子" else round(v, 2))
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out or None


def _sweep_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def sweep_generate(editor_value, ax_param, ax_vals, ay_param, ay_vals):
    if ax_param == ay_param:
        return None, "横轴和纵轴别选同一个参数"
    arr = extract_array(editor_value)
    mask = extract_ink(arr) if arr is not None else None
    if mask is None or int((mask > 64).sum()) < 40:
        return None, "先在画布上画一个轮廓"
    xs = _sweep_parse(ax_vals, ax_param)
    ys = _sweep_parse(ay_vals, ay_param)
    if not xs or not ys:
        return None, "参数取值格式：逗号分隔，如 0.8,1.2,1.6"
    total = len(xs) * len(ys)
    if total > 24:
        return None, f"组合 {len(xs)}×{len(ys)} = {total} 张，太多（上限 24 张）"
    if not os.path.exists(os.path.join(LORA_DIR, "pytorch_lora_weights.safetensors")):
        return None, "找不到 LoRA 权重（models/naiwa_lora/out）"
    t0 = time.time()
    pipe = get_lora_pipe()
    g = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LINEAR)
    control = Image.fromarray(np.stack([255 - g] * 3, -1))
    base = {"LoRA 强度": 1.4, "步数": 10, "引导": 3.5, "种子": 1234}
    grid = []
    for yv in ys:
        row = []
        for xv in xs:
            p = dict(base)
            p[ax_param] = xv
            p[ay_param] = yv
            pipe.set_adapters(["naiwa", "lcm"], adapter_weights=[float(p["LoRA 强度"]), 1.0])
            gen = torch.Generator(device=DEVICE).manual_seed(int(p["种子"]))
            img = pipe(
                LORA_PROMPT, negative_prompt=LORA_NEG, image=control,
                num_inference_steps=int(p["步数"]), guidance_scale=float(p["引导"]),
                generator=gen,
            ).images[0]
            row.append(img)
        grid.append(row)
    dt = time.time() - t0
    cell, lbl = 256, 24
    f = _sweep_font(16)
    W = 118 + len(xs) * cell + 8
    H = lbl + len(ys) * cell + 26
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, row in enumerate(grid):
        for i, im in enumerate(row):
            sheet.paste(im.resize((cell - 6, cell - 6), Image.LANCZOS), (118 + i * cell + 3, lbl + j * cell + 3))
    for i, xv in enumerate(xs):
        t = str(xv)
        d.text((118 + i * cell + (cell - d.textlength(t, font=f)) / 2, 4), t, font=f, fill=(0, 0, 0))
    for j, yv in enumerate(ys):
        d.text((8, lbl + j * cell + cell / 2 - 10), str(yv), font=f, fill=(0, 0, 0))
    d.text((8, H - 20), f"x={ax_param} / y={ay_param}", font=_sweep_font(14), fill=(90, 90, 90))
    out_dir = os.path.join(HERE, "data_raw", "naiwa_img2img")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sweep_ui_{time.strftime('%Y%m%d_%H%M%S')}.png")
    sheet.save(path)
    info = (f"{ax_param}({len(xs)}) × {ay_param}({len(ys)}) = {total} 张 | 耗时 {dt:.1f}s | "
            f"其余固定：强度1.4/步数10/引导3.5/种子1234 | 已存 {path}")
    return sheet, info
