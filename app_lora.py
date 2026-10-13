"""LoRA 版 (SD1.5 + 奶娃LoRA + LCM + ControlNet 512px)：画板 / 图 / 视频。

- 画板: 画任意颜色轮廓 -> HED 风格边缘 -> ControlNet 生成
- 图: 上传 -> rembg抠主体 -> HED边缘 -> ControlNet 生成
- 视频: 逐帧抠主体+HED -> ControlNet 生成（限 8 秒 / 96 帧）
- 无 img2img；种子可固定或真随机

Run: python app_lora.py     ->  http://127.0.0.1:7864
"""

import glob
import os

import numpy as np
from PIL import Image

import gradio as gr

import app_naiwa as core

PORT = 7864
BLACK = np.zeros((512, 512, 3), np.uint8)

_last_pick = None


def random_sample():
    global _last_pick
    files = glob.glob(os.path.join(core.TEST_DIR, "*.jpg"))
    if not files:
        return gr.update()
    pick = np.random.choice(files)
    for _ in range(5):
        if len(files) == 1 or pick != _last_pick:
            break
        pick = np.random.choice(files)
    _last_pick = pick
    pair = np.array(Image.open(pick).convert("RGB"))
    return pair[:, : pair.shape[1] // 2]


def sketch_generate(editor_value, steps, lora_w, guidance, seed, true_random):
    return core.lora_generate_sketch(editor_value, steps, lora_w, guidance, seed, true_random)


def image_generate(image, steps, lora_w, guidance, seed, true_random):
    return core.lora_generate_image(image, steps, lora_w, guidance, seed, true_random)


def video_generate(video, steps, lora_w, guidance, seed, true_random):
    return core.lora_generate_video(video, steps, lora_w, guidance, seed, true_random)


with gr.Blocks(title="Sketch2Naiwa LoRA 版") as demo:
    gr.Markdown(
        "# 🧬 Sketch2Naiwa — LoRA 版（512px）\n"
        "画板 / 图 / 视频 三个入口，全部 **抠边缘 → ControlNet** 生成（无 img2img）。\n"
        "自训 LoRA（740 张素材）+ LCM 少步采样；**真随机** 打勾时每次换随机种子。"
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
                label="画布（笔色任意取 / 橡皮 / 撤销由画板工具栏提供）",
            )
            with gr.Row():
                bg_picker = gr.ColorPicker(value="#000000", label="背景色")
                bg_btn = gr.Button("新建画布（应用背景色）")
            with gr.Row():
                gen_btn = gr.Button("生成奶娃（ControlNet）", variant="primary")
                rand_btn = gr.Button("随机奶娃轮廓")
            with gr.Row():
                steps_sl = gr.Slider(minimum=4, maximum=12, step=1, value=10, label="步数")
                lora_w_sl = gr.Slider(minimum=0.6, maximum=2.4, step=0.1, value=1.4, label="LoRA 强度")
                guide_sl = gr.Slider(minimum=1.0, maximum=4.5, step=0.5, value=3.5, label="引导")
            with gr.Row():
                true_rand_cb = gr.Checkbox(value=True, label="真随机种子（每次生成都不同）")
                seed_num = gr.Number(value=1234, label="固定种子（取消勾选后用）", precision=0)
        with gr.Column(scale=1):
            out_img = gr.Image(label="奶娃", height=460, width=460)
            out_info = gr.Textbox(label="状态", interactive=False, lines=2)

    gr.Markdown("---\n## 🔬 参数矩阵扫描 — 同一张画 × 两组参数全组合\n"
                "选两个参数各自填取值（逗号分隔），一次生成全部组合成一张对比大图（固定种子 1234）。上限 24 张。")
    with gr.Row():
        with gr.Column(scale=1):
            sweep_x = gr.Dropdown(choices=core.SWEEP_PARAMS, value="LoRA 强度", label="横轴参数")
            sweep_xv = gr.Textbox(value="0.8, 1.2, 1.6, 2.0", label="横轴取值（逗号分隔）")
            sweep_y = gr.Dropdown(choices=core.SWEEP_PARAMS, value="步数", label="纵轴参数")
            sweep_yv = gr.Textbox(value="4, 8, 12", label="纵轴取值（逗号分隔）")
            sweep_btn = gr.Button("生成参数矩阵", variant="primary")
            sweep_info = gr.Textbox(label="状态", interactive=False, lines=2)
        with gr.Column(scale=2):
            sweep_out = gr.Image(label="对比矩阵（横轴→ / 纵轴↓）", height=640)

    gr.Markdown("---\n## 🖼 图 — 上传图片，抠主体 + HED 边缘 → ControlNet")
    with gr.Row():
        with gr.Column(scale=1):
            pic_in = gr.Image(type="numpy", label="① 上传图片", height=300)
            pic_btn = gr.Button("抠主体 + 提取边缘 + 生成奶娃", variant="primary")
            pic_info = gr.Textbox(label="状态", interactive=False, lines=2)
        with gr.Column(scale=1):
            pic_cut = gr.Image(label="② 抠出的主体（贴白底）", height=300)
        with gr.Column(scale=1):
            pic_edge = gr.Image(label="③ 提取的边缘（ControlNet 输入）", height=300)
        with gr.Column(scale=1):
            pic_out = gr.Image(label="④ 奶娃", height=300)

    gr.Markdown("---\n## 🎬 视频 — 逐帧生成奶娃\n最多处理前 8 秒 / 96 帧（12fps 采样），同一视频内种子固定保持帧间稳定。")
    with gr.Row():
        with gr.Column(scale=1):
            vid_in = gr.File(label="① 上传视频（mp4 等）", file_types=["video"])
            vid_btn = gr.Button("逐帧生成奶娃视频", variant="primary")
            vid_info = gr.Textbox(label="状态", interactive=False, lines=3)
        with gr.Column(scale=1):
            vid_cut = gr.Video(label="② 抠主体视频（贴白底）", height=300)
        with gr.Column(scale=1):
            vid_out = gr.Video(label="③ 奶娃视频", height=300)

    gen_btn.click(sketch_generate, [canvas, steps_sl, lora_w_sl, guide_sl, seed_num, true_rand_cb], [out_img, out_info])
    rand_btn.click(random_sample, None, [canvas])
    bg_btn.click(core.new_canvas, [bg_picker], [canvas])
    sweep_btn.click(core.sweep_generate, [canvas, sweep_x, sweep_xv, sweep_y, sweep_yv], [sweep_out, sweep_info])
    pic_btn.click(image_generate, [pic_in, steps_sl, lora_w_sl, guide_sl, seed_num, true_rand_cb], [pic_cut, pic_edge, pic_out, pic_info])
    vid_btn.click(video_generate, [vid_in, steps_sl, lora_w_sl, guide_sl, seed_num, true_rand_cb], [vid_cut, vid_out, vid_info])

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("NAIWA_HOST", "0.0.0.0"),
        server_port=PORT,
        inbrowser=True,
        share=os.environ.get("NAIWA_SHARE") == "1",
    )
