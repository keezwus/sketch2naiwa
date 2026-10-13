"""演示版：只保留画板生图 + 模型选择（两个最优权重），面向十几人并发。

- LoRA 512px：SD1.5 + 奶娃 LoRA v3（328 张）+ LCM + ControlNet-scribble（精细，约 2 秒/张）
- pix2pix 256px：naiwa_pix2pix2 epoch180（极速，约 14 毫秒/张）

并发：Gradio 队列串行执行（单卡 GPU），多人同时点生成会自动排队，先到先得。

Run: python app_demo.py  ->  http://127.0.0.1:7870
"""

import base64
import os

import numpy as np

import gradio as gr

import app_naiwa as core

PORT = 7870
BLACK = np.zeros((512, 512, 3), np.uint8)

LORA_CHOICE = "LoRA 512px 精细版（推荐）"
PIX2_CHOICE = "pix2pix 256px 极速版"
LORA_STEPS, LORA_W, LORA_GUIDE = 10, 1.4, 3.5

_pix2_ckpt = None


def _bgm_html():
    """Hidden one-shot sound; played each time 生成奶娃 is clicked (see gen_btn js)."""
    try:
        with open(os.path.join(core.HERE, "assets", "naiwaSound.m4a"), "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return ""
    return f'<audio id="naiwa-bgm" preload="auto" src="data:audio/mp4;base64,{b64}"></audio>'


def generate(canvas, model_type):
    if os.environ.get("NAIWA_DEBUG") == "1":
        print("[debug] canvas:", type(canvas), repr(canvas)[:300], flush=True)
    try:
        if model_type == LORA_CHOICE:
            return core.lora_generate_sketch(canvas, LORA_STEPS, LORA_W, LORA_GUIDE, 1234, True)
        choice = core.find_choice("epoch 180") or core.default_choice()
        return core.pix2_generate(canvas, choice, gacha=True)
    except Exception as e:
        return None, f"生成失败：{e!r}"


with gr.Blocks(title="Sketch2Naiwa 演示") as demo:
    gr.Markdown(
        "# 画个轮廓，变奶娃\n"
        "在画板上随手画一个形状，点 **生成奶娃**。"
        "两个模型都可以试试，人多了会自动排队。"
    )
    with gr.Row():
        with gr.Column(scale=1):
            canvas = gr.Sketchpad(
                value=BLACK,
                brush=gr.Brush(
                    colors=["#FFFFFF", "#000000", "#FF5252", "#42A5F5", "#66BB6A", "#FFCA28"],
                    default_color="#FFFFFF",
                    color_mode="defaults",
                ),
                canvas_size=(512, 512),
                height=460,
                width=460,
                type="numpy",
                label="画布",
            )
            with gr.Row():
                clear_btn = gr.Button("清空画布（黑底）")
            model_radio = gr.Radio(
                choices=[LORA_CHOICE, PIX2_CHOICE],
                value=LORA_CHOICE,
                label="模型",
            )
            gen_btn = gr.Button("生成奶娃", variant="primary", size="lg")
            gr.HTML(_bgm_html())
        with gr.Column(scale=1):
            out_img = gr.Image(label="奶娃", height=512, width=512)
            out_info = gr.Textbox(label="状态", interactive=False, lines=2)

    gen_btn.click(generate, [canvas, model_radio], [out_img, out_info])
    gen_btn.click(
        None, None, None,
        js=("() => { const a = document.getElementById('naiwa-bgm'); "
            "if (!a) return; a.currentTime = 0; a.volume = 0.5; "
            "a.play().catch(() => {}); }"),
    )
    clear_btn.click(lambda: core.new_canvas("#000000"), None, [canvas], queue=False)

if __name__ == "__main__":
    print("[demo] 预加载 pix2pix epoch180 ...", flush=True)
    _pix2_ckpt = core.find_choice("epoch 180") or core.default_choice()
    core.load_generator(_pix2_ckpt)
    print("[demo] 预加载 LoRA v3 ...", flush=True)
    core.get_lora_pipe()
    print("[demo] 模型就绪，启动服务", flush=True)
    demo.queue(default_concurrency_limit=1, max_size=64).launch(
        server_name=os.environ.get("NAIWA_HOST", "0.0.0.0"),
        server_port=PORT,
        inbrowser=os.environ.get("NAIWA_NO_BROWSER") != "1",
        show_error=False,
    )
