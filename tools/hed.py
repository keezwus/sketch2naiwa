"""Vendored HED edge detector (ControlNetHED, Apache-2.0, from lllyasviel/ControlNet annotator/hed)."""

import os
import time

import cv2
import numpy as np
import requests
import torch
from einops import rearrange

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_raw", "hed")
CKPT_PATH = os.path.abspath(os.path.join(CKPT_DIR, "ControlNetHED.pth"))
CKPT_URLS = [
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth",
    "https://hf-mirror.com/lllyasviel/Annotators/resolve/main/ControlNetHED.pth",
]


def download_ckpt():
    os.makedirs(CKPT_DIR, exist_ok=True)
    if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 29_000_000:
        return CKPT_PATH
    for attempt in range(3):
        for url in CKPT_URLS:
            try:
                have = os.path.getsize(CKPT_PATH) if os.path.exists(CKPT_PATH) else 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                mode = "ab" if have else "wb"
                print(f"Downloading HED checkpoint from {url} (attempt {attempt + 1})", flush=True)
                with requests.get(url, stream=True, timeout=(15, 60), headers=headers) as r:
                    if have > 0 and r.status_code == 200:
                        have = 0
                        mode = "wb"
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0)) + have
                    done = have
                    t0 = time.time()
                    tlast = t0
                    with open(CKPT_PATH, mode) as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                            done += len(chunk)
                            now = time.time()
                            if now - tlast >= 3:
                                speed = (done - have) / max(now - t0, 1e-6) / 1e6
                                pct = f"{100 * done / total:.0f}%" if total else "?"
                                print(f"  {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}) {speed:.1f} MB/s", flush=True)
                                tlast = now
                if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 29_000_000:
                    print(f"HED checkpoint OK: {os.path.getsize(CKPT_PATH) / 1e6:.1f} MB", flush=True)
                    return CKPT_PATH
            except Exception as e:
                print("failed:", repr(e), flush=True)
    raise RuntimeError("Could not download HED checkpoint")


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(input_channel, output_channel, 3, 1, 1))
        for _ in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(output_channel, output_channel, 3, 1, 1))
        self.projection = torch.nn.Conv2d(output_channel, 1, 1, 1, 0)

    def forward(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = torch.nn.functional.relu(conv(h))
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(3, 64, 2)
        self.block2 = DoubleConvBlock(64, 128, 2)
        self.block3 = DoubleConvBlock(128, 256, 3)
        self.block4 = DoubleConvBlock(256, 512, 3)
        self.block5 = DoubleConvBlock(512, 512, 3)

    def forward(self, x):
        h = x - self.norm
        h, p1 = self.block1(h)
        h, p2 = self.block2(h, down_sampling=True)
        h, p3 = self.block3(h, down_sampling=True)
        h, p4 = self.block4(h, down_sampling=True)
        h, p5 = self.block5(h, down_sampling=True)
        return p1, p2, p3, p4, p5


class HEDDetector:
    def __init__(self, device="cuda"):
        modelpath = download_ckpt()
        self.device = device
        self.net = ControlNetHED_Apache2().to(device).float().eval()
        state = torch.load(modelpath, map_location="cpu")
        self.net.load_state_dict(state)

    def __call__(self, rgb_image):
        """rgb_image: HxWx3 uint8 RGB. Returns HxW uint8 edge map."""
        h, w, _ = rgb_image.shape
        with torch.no_grad():
            x = torch.from_numpy(rgb_image.copy()).float().to(self.device)
            x = rearrange(x, "h w c -> 1 c h w")
            edges = self.net(x)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge
