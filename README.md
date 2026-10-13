# Sketch2Naiwa

**简笔画 / 照片 / 视频 / 手机直播 → 奶娃角色生成** · 课程项目最终提交仓库

本仓库是一个“开箱即跑”的完整交付：代码 + 两条产品线的最优模型 + 一键启动脚本 + 课程报告全部素材。
所有模型推理在单台 8GB 显存笔记本上运行，无需联网（首次除外）。

---

## 仓库内容

```
sketch2naiwa/
├── 启动-演示版.bat           # 一键：画板生图 + 模型选择（课堂演示用，十几人并发排队）
├── 启动-完整版.bat           # 一键：pix2pix 完整版(7861) + LoRA 完整版(7864)
├── 启动-实时直播.bat         # 一键：手机摄像头 → 电脑实时变奶娃(7863)
│
├── app_demo.py               # 演示版（精简：画板 + 双模型切换）
├── app_pix2.py               # pix2pix 完整版（画板/图/视频）
├── app_lora.py               # LoRA 完整版（画板/图/视频/参数矩阵扫描）
├── app_naiwa.py              # 核心库（模型加载、抠图、边缘、生成管线）
├── serve_naiwa.py            # 双 App 启动器
│
├── models/                   # 模型目录
│   ├── naiwa_lora/out/       # ★ 产品线 B：自训 LoRA（v3 checkpoint-200，12MB）
│   ├── sd15-base/            #   基座 SD1.5（fp16，第三方）
│   ├── controlnet-scribble/  #   线稿控制模型（第三方）
│   ├── lcm-lora-sd15/        #   少步采样蒸馏适配器（第三方）
│   ├── taesd/                #   微型 VAE（直播加速）
│   └── rembg/                #   抠图模型（离线预置）
├── pytorch-CycleGAN-and-pix2pix/
│   ├── models/               #   pix2pix 网络定义
│   ├── checkpoints/naiwa_pix2pix2/180_net_G.pth   # ★ 产品线 A：自训 pix2pix（207MB）
│   └── datasets/naiwa_final/test/                 #   示例轮廓（“随机轮廓”按钮用）
├── weights/                  # 展示用权重副本（LoRA v3 单一文件）
├── data_raw/hed/             # HED 边缘检测权重（离线预置）
├── tools/                    # HED 检测器 / 直播服务端 / 直播电脑端显示
├── setup/                    # 基础模型下载脚本（如换机器丢失 models 时用）
└── docs/
    ├── 课程报告材料.md        # ★ 报告七板块全部素材（环境/数据/应用/动机/挑战/效果/心得）
    ├── 技术讲解与答辩.md      # ★ 技术问答准备（技术卡片 + 高频问答 + 演示流程）
    ├── 开发日志.md            # 全流程开发记录（问题与决策时间线）
    └── images/               # 报告用图（评估、对比、修复前后）
```

## 一键启动

| 脚本 | 内容 | 地址 |
| --- | --- | --- |
| `启动-演示版.bat` | 画板 + 模型二选一（LoRA 512px / pix2pix 256px） | http://127.0.0.1:7870 |
| `启动-完整版.bat` | pix2pix 完整版 + LoRA 完整版（画板/图片/视频/扫描） | http://127.0.0.1:7861 、 :7864 |
| `启动-实时直播.bat` | 手机拍 → 电脑大屏实时变奶娃 | 手机访问 `https://<本机IP>:7863` |

说明：
- 双击运行即可，脚本会自动设置环境变量并打开浏览器；首次加载模型约 1 分钟；
- 局域网内其他电脑/手机直接访问 `http://<本机IP>:7870` 等端口；
- 实时直播为自签 HTTPS 证书，手机浏览器提示“不安全”时选择“继续访问”。

## 从 GitHub 克隆后如何运行（同学 / 换机指引）

```bash
git clone https://github.com/keezwus/sketch2naiwa.git
cd sketch2naiwa
pip install -r requirements.txt
# torch 按显卡装 CUDA 版（本项目用 cu129）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
python setup/download_all.py   # 自动补齐：pix2pix 权重(207MB) + SD1.5 基础栈(~4GB)
```

然后双击 `启动-演示版.bat`（或 `启动-完整版.bat` / `启动-实时直播.bat`）即可。

要点：
- 207MB 的 pix2pix 权重不在 git 仓库内，`download_all.py` 会自动从
  [Release v1.0](https://github.com/keezwus/sketch2naiwa/releases/tag/v1.0) 下载（断点续传）；
- SD1.5 基础栈走 hf-mirror 镜像下载（约 4GB），HED 边缘/抠图模型首次使用时自动补下；
- 国内网络拉 GitHub 大文件慢时记得开代理；
- 如果拿到的是“完整文件夹版”（U 盘/局域网拷贝），所有模型已内置，双击脚本即可，无需任何下载。

## 两条产品线（最优模型）

| | 产品线 A · 极速版 | 产品线 B · 精细版 |
| --- | --- | --- |
| 模型 | 自训 pix2pix（U-Net + PatchGAN） | SD1.5 + 自训 LoRA + ControlNet + LCM |
| 权重 | `pytorch-CycleGAN-and-pix2pix/checkpoints/naiwa_pix2pix2/180_net_G.pth` | `models/naiwa_lora/out/pytorch_lora_weights.safetensors` |
| 数据 | 2358 训练对（328 张素材 × 边缘提取 × 9 倍增强） | 328 张精选素材（80% 留白 + 文本描述） |
| 分辨率 | 256px | 512px |
| 速度 | ≈14 ms/张 | ≈1.2~2.7 s/张 |
| 场景 | 视频逐帧、实时直播 | 画板、单图精修 |

两个模型均训练自本仓库 328 张人工清洗后的素材集；详细训练过程见 `docs/开发日志.md`。

## 环境依赖

- Windows + NVIDIA GPU（本项目在 RTX 5060 Laptop 8GB 上测试）
- Python 3.13，依赖见 `requirements.txt`（torch 需按 CUDA 版本单独安装）
- 如 `models/` 中第三方基础模型缺失，运行 `python setup/download_all.py` 重新下载
  （含 Release 权重与 hf-mirror 基础栈，断点续传）

## 文档索引

- **写课程报告** → `docs/课程报告材料.md`（七板块素材 + 图表清单）
- **技术问答准备** → `docs/技术讲解与答辩.md`（10 张技术卡片 + 10 条高频问答 + 演示流程）
- **开发全过程** → `docs/开发日志.md`（时间线：数据事故、面部问题、背景修复等）

## 许可与致谢

- 训练数据为网络开源梗图素材，仅用于课程学习；
- 基座模型：Stable Diffusion 1.5、ControlNet（lllyasviel）、LCM-LoRA、
  HED（ControlNetHED）、U²-Net（rembg）、pix2pix（junyanz/pytorch-CycleGAN-and-pix2pix），
  遵循各自开源许可（Apache-2.0 / CreativeML Open RAIL 等）。
