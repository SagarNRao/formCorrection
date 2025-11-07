@echo off
REM ==== EDIT IF NEEDED ====
set WEBCAM=Lenovo FHD Webcam
set IP=10.227.207.170
REM =======================

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
  -bufsize 1200k ^
  -fflags nobuffer+flush_packets ^
  -flags low_delay ^
  -probesize 32 ^
  -analyzeduration 0 ^
  -rtsp_transport tcp ^
  -f rtsp ^
  rtsp://%IP%:8554/webcam