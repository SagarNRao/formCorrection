@echo off
set WEBCAM=Lenovo FHD Webcam
set SERVER=127.0.0.1
set PATH_NAME=webcam

echo.
echo === KILLING CONFLICTING APPS ===
taskkill /f /im Camera.exe 2>nul
taskkill /f /im Zoom.exe 2>nul
taskkill /f /im Teams.exe 2>nul
taskkill /f /im msedge.exe 2>nul
taskkill /f /im chrome.exe 2>nul
echo.

echo === STARTING FFMPEG PUSH TO MEDIAMTX ===
ffmpeg ^
  -f dshow ^
  -i video="%WEBCAM%" ^
  -c:v libx264 ^
  -preset ultrafast ^
  -tune zerolatency ^
  -profile:v baseline ^
  -pix_fmt yuv420p ^
  -s 640x480 ^
  -r 15 ^
  -g 10 ^
  -b:v 600k ^
  -f rtsp ^
  rtsp://%SERVER%:8554/%PATH_NAME%