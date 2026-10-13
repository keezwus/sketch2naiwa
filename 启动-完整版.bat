@echo off
chcp 65001 >nul
cd /d %~dp0
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
set REMBG_HOME=%cd%\models\rembg
echo [完整版] 双 App：pix2pix 完整版(7861) + LoRA 完整版(7864)
echo 画板 / 图片 / 视频 / 参数矩阵扫描，浏览器会自动打开
python serve_naiwa.py
pause
