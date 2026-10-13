@echo off
chcp 65001 >nul
cd /d %~dp0
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
set REMBG_HOME=%cd%\models\rembg
echo [实时直播] 手机摄像头 -^> 电脑实时变奶娃
echo 启动后手机浏览器打开：https://本机局域网IP:7863 （证书警告选"继续访问"）
start "Sketch2Naiwa Live Server" python tools\live_web_lora.py
timeout /t 10 /nobreak >nul
python tools\live_view_lora.py
pause
