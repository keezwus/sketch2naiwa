@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 echo [错误] 未找到 python 命令，请先安装 Python 并勾选 Add to PATH
if errorlevel 1 pause
if errorlevel 1 exit /b 1
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
set REMBG_HOME=%cd%\models\rembg
echo [完整版] 双 App：pix2pix 完整版 7861 + LoRA 完整版 7864
echo 画板 / 图片 / 视频 / 参数矩阵扫描，浏览器会自动打开
python serve_naiwa.py
pause
