"""Download SD1.5 + ControlNet-scribble + LCM-LoRA into models/ (hf-mirror, resumable)."""

import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://hf-mirror.com"
LOG = os.path.join(ROOT, "models", "sd15_stack_download.log")

REPOS = [
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", os.path.join(ROOT, "models", "sd15-base"), [
        "model_index.json", "scheduler/scheduler_config.json",
        "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    ]),
    ("lllyasviel/control_v11p_sd15_scribble", os.path.join(ROOT, "models", "controlnet-scribble"), [
        "config.json", "diffusion_pytorch_model.safetensors",
    ]),
    ("latent-consistency/lcm-lora-sdv1-5", os.path.join(ROOT, "models", "lcm-lora-sd15"), [
        "pytorch_lora_weights.safetensors",
    ]),
]


def log(m):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def main():
    log("START " + time.strftime("%H:%M:%S"))
    grand = 0
    for repo, local, wanted in REPOS:
        os.makedirs(local, exist_ok=True)
        tree = requests.get(f"{BASE}/api/models/{repo}/tree/main?recursive=true", timeout=30).json()
        files = [(f["path"], f["size"]) for f in tree if f["type"] == "file" and f["path"] in wanted]
        total = sum(s for _, s in files)
        log(f"[{repo}] {len(files)} files, {total / 1e9:.2f} GB")
        done = 0
        for path, size in files:
            dest = os.path.join(local, path.replace("/", os.sep))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            if have >= size and size > 0:
                done += size
                grand += size
                log(f"SKIP {repo}/{path}")
                continue
            url = f"{BASE}/{repo}/resolve/main/{path}"
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as r:
                if have and r.status_code == 200:
                    have = 0
                    mode = "wb"
                elif have and r.status_code == 206:
                    mode = "ab"
                else:
                    have = 0
                    mode = "wb"
                r.raise_for_status()
                with open(dest, mode) as f:
                    tlast = time.time()
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        grand += len(chunk)
                        now = time.time()
                        if now - tlast >= 6:
                            log(f"  [{repo}] {done / 1e9:.2f}/{total / 1e9:.2f} GB (grand {grand / 1e9:.2f})")
                            tlast = now
            log(f"OK {repo}/{path}")
    log("ALL DONE")


if __name__ == "__main__":
    main()
