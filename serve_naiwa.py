"""Headless launcher for the two split Naiwa apps (no browser popup).

  app_pix2  ->  port 7861   原版 pix2pix
  app_lora  ->  port 7864   LoRA ControlNet 版

Env:
  NAIWA_HOST  bind address (default 0.0.0.0 = LAN accessible; use 127.0.0.1 for local-only)
  NAIWA_SHARE =1 to request a public gradio.live tunnel (for phones outside the LAN)
"""

import os
import threading

import app_lora
import app_pix2

if __name__ == "__main__":
    host = os.environ.get("NAIWA_HOST", "0.0.0.0")
    share = os.environ.get("NAIWA_SHARE") == "1"
    inb = os.environ.get("NAIWA_NO_BROWSER") != "1"
    print(f"pix2:  http://127.0.0.1:{app_pix2.PORT}", flush=True)
    print(f"lora:  http://127.0.0.1:{app_lora.PORT}", flush=True)
    app_pix2.demo.launch(server_name=host, server_port=app_pix2.PORT, inbrowser=inb, share=share, prevent_thread_lock=True)
    app_lora.demo.launch(server_name=host, server_port=app_lora.PORT, inbrowser=inb, share=share)
