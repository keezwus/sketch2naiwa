@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 echo [错误] 未找到 python 命令，请先安装 Python 并勾选 Add to PATH
if errorlevel 1 pause
if errorlevel 1 exit /b 1
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
set REMBG_HOME=%cd%\models\rembg
echo [演示版] 画板生图 + 模型选择（LoRA 512px / pix2pix 256px）
echo 正在加载模型（首次约 1 分钟），完成后浏览器会自动打开 http://127.0.0.1:7870
python app_demo.py
pause
