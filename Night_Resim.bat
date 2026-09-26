@echo off
cd /d "%~dp0"
title DeltaZero - night re-simulation (download + simulate, keeps going through errors)
rem Start_DeltaZero.bat must not run at the same time. Uses every running helper instance.
rem Downloads the most awarded TMX maps and their 5 fastest replays while re-simulating;
rem maps already done are skipped, failing maps are logged and skipped, a lost game connection
rem is retried every 15 s. Progress: data\resim\night_status.json (errors: night_errors.jsonl).
set HOURS=
set /p HOURS=How many hours (empty = 8): 
if "%HOURS%"=="" set HOURS=8
set DRAW=
set /p DRAW=Turn off rendering in the game(s) for speed? (y/N): 
set NODRAW=
if /i "%DRAW%"=="y" set NODRAW=--no-draw
python tmdriver.py %NODRAW% night --hours %HOURS%
pause
