"""原版 pix2pix 版 (256px)：画板 / 图 / 视频。

- 画板: 画任意颜色轮廓 -> 生成 (抽奖模式=每次随机微变)
- 图: 上传 -> rembg抠主体 -> HED边缘 -> 生成
- 视频: 逐帧抠主体+HED -> 生成

Run: python app_pix2.py     ->  http://127.0.0.1:7861
"""

import glob
import os

import numpy as np
from PIL import Image

import gradio as gr

import app_naiwa as core

PORT = 7861
BLACK = np.zeros((512, 512, 3), np.uint8)

_last_pick = None


def random_sample():
    global _last_pick
    files = glob.glob(os.path.join(core.TEST_DIR, "*.jpg"))
    if not files:
        return gr.update(), gr.update()
    pick = np.random.choice(files)
    for _ in range(5):
        if len(files) == 1 or pick != _last_pick:
            break
        pick = np.random.choice(files)
    _last_pick = pick
    pair = np.array(Image.open(pick).convert("RGB"))
    return pair[:, : pair.shape[1] // 2], gr.update(value=core.find_choice("latest", "奶娃") or core.default_choice())


def shoe_example():
    return core.shoe_outline(), gr.update(value=core.find_choice("edges2shoes") or core.default_choice())


def generate(editor_value, choice, gacha):
    return core.pix2_generate(editor_value, choice, gacha)


def picto_generate(image, choice, gacha):
    return core.pix2_generate_image(image, choice, gacha)


def video_generate(video, choice):
    return core.pix2_generate_video(video, choice)


def refresh_models(cur):
    choices = core.model_choices()
    val = cur if cur in choices else core.default_choice()
    epochs = core.epoch_list()
    status = f"模型存档: {len(epochs)} 个（最高 epoch {max(epochs) if epochs else 0} / {core.TOTAL_EPOCHS}）"
    return gr.update(choices=choices, value=val), status


with gr.Blocks(title="Sketch2Naiwa 原版 pix2pix") as demo:
    gr.Markdown(
        "# 🐈 Sketch2Naiwa — 原版 pix2pix（256px）\n"
        "画板 / 图 / 视频 三个入口，全部走自训 pix2pix（从 0 训练 240 epoch）。\n"
        "画板支持任意笔色/背景色；「抽奖模式」每次生成随机微形变（真随机）。"
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
                gen_btn = gr.Button("生成奶娃", variant="primary")
                rand_btn = gr.Button("随机奶娃轮廓")
                shoe_btn = gr.Button("载入鞋示例轮廓")
            gacha_cb = gr.Checkbox(value=True, label="抽奖模式：每次生成随机微变化（真随机）")
            model_dd = gr.Dropdown(choices=core.model_choices(), value=core.default_choice(), label="模型")
            refresh_btn = gr.Button("刷新模型列表 / 状态")
        with gr.Column(scale=1):
            out_img = gr.Image(label="奶娃", height=460, width=460)
            out_info = gr.Textbox(label="状态", interactive=False, lines=2)

    gr.Markdown("---\n## 🖼 图 — 上传图片，抠主体 + HED 边缘生成")
    with gr.Row():
        with gr.Column(scale=1):
            pic_in = gr.Image(type="numpy", label="① 上传图片", height=300)
            pic_btn = gr.Button("抠主体 + 提取边缘 + 生成奶娃", variant="primary")
            pic_info = gr.Textbox(label="状态", interactive=False, lines=2)
        with gr.Column(scale=1):
            pic_cut = gr.Image(label="② 抠出的主体（贴白底）", height=300)
        with gr.Column(scale=1):
            pic_edge = gr.Image(label="③ 提取的边缘（模型输入）", height=300)
        with gr.Column(scale=1):
            pic_out = gr.Image(label="④ 奶娃", height=300)

    gr.Markdown("---\n## 🎬 视频 — 逐帧生成奶娃\n最多处理前 15 秒 / 300 帧（12fps 采样），不做随机形变保持帧间稳定。")
    with gr.Row():
        with gr.Column(scale=1):
            vid_in = gr.File(label="① 上传视频（mp4 等）", file_types=["video"])
            vid_btn = gr.Button("逐帧生成奶娃视频", variant="primary")
            vid_info = gr.Textbox(label="状态", interactive=False, lines=3)
        with gr.Column(scale=1):
            vid_cut = gr.Video(label="② 抠主体视频（贴白底）", height=300)
        with gr.Column(scale=1):
            vid_out = gr.Video(label="③ 奶娃视频", height=300)

    gen_btn.click(generate, [canvas, model_dd, gacha_cb], [out_img, out_info])
    rand_btn.click(random_sample, None, [canvas, model_dd])
    shoe_btn.click(shoe_example, None, [canvas, model_dd])
    bg_btn.click(core.new_canvas, [bg_picker], [canvas])
    refresh_btn.click(refresh_models, model_dd, [model_dd, out_info])
    pic_btn.click(picto_generate, [pic_in, model_dd, gacha_cb], [pic_cut, pic_edge, pic_out, pic_info])
    vid_btn.click(video_generate, [vid_in, model_dd], [vid_cut, vid_out, vid_info])
    demo.load(refresh_models, model_dd, [model_dd, out_info])

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("NAIWA_HOST", "0.0.0.0"),
        server_port=PORT,
        inbrowser=True,
        share=os.environ.get("NAIWA_SHARE") == "1",
    )
