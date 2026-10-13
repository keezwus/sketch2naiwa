"""一键补齐运行所需模型（在克隆的新机器上使用，断点续传）。

1) 产品线 A 权重（pix2pix epoch180，207MB）-> GitHub Release v1.0 附件
2) 第三方基础栈（SD1.5 + ControlNet + LCM，约 4GB）-> hf-mirror 镜像

用法:
    python setup/download_all.py
"""

import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PIX2_DST = os.path.join(ROOT, "pytorch-CycleGAN-and-pix2pix", "checkpoints", "naiwa_pix2pix2", "180_net_G.pth")
PIX2_URL = "https://github.com/keezwus/sketch2naiwa/releases/download/v1.0/naiwa_pix2pix2_epoch180_net_G.pth"
PIX2_MIN = 200_000_000


def download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    for attempt in range(3):
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 180), allow_redirects=True) as r:
                if have and r.status_code == 200:
                    have, mode = 0, "wb"
                elif have and r.status_code == 206:
                    mode = "ab"
                else:
                    have, mode = 0, "wb"
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + have
                done = have
                with open(dst, mode) as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if done % (25 * 1024 * 1024) < 1024 * 1024:
                            print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
            return
        except Exception as e:
            print(f"  下载出错（第 {attempt + 1} 次重试）：{e!r}", flush=True)
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
    raise RuntimeError(f"下载失败: {url}")


def main():
    print("== 1/2 产品线 A：pix2pix 权重（207MB，GitHub Release）==")
    if os.path.exists(PIX2_DST) and os.path.getsize(PIX2_DST) >= PIX2_MIN:
        print("已存在，跳过:", PIX2_DST)
    else:
        download(PIX2_URL, PIX2_DST)
        print("OK:", PIX2_DST)

    print("== 2/2 第三方基础栈：SD1.5 + ControlNet + LCM（约 4GB，hf-mirror）==")
    sys.path.insert(0, HERE)
    import download_sd15_stack

    download_sd15_stack.main()
    print()
    print("全部完成。双击「启动-演示版.bat」等脚本即可运行。")


if __name__ == "__main__":
    main()
