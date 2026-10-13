@echo off
chcp 65001 >nul
cd /d %~dp0
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
set REMBG_HOME=%cd%\models\rembg
echo [演示版] 画板生图 + 模型选择（LoRA 512px / pix2pix 256px）
echo 正在加载模型（首次约 1 分钟），完成后浏览器会自动打开 http://127.0.0.1:7870
python app_demo.py
pause
