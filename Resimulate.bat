@echo off
cd /d "%~dp0"
title DeltaZero - Re-simulate replays (batch mode)
rem Start_DeltaZero.bat must not run at the same time (one Python connection per game).
rem Uses every running helper instance; replays already re-simulated are skipped.
set HOURS=
set /p HOURS=How many hours (empty = 3): 
if "%HOURS%"=="" set HOURS=3
python tmdriver.py resim --source bulk --replays 5 --hours %HOURS%
pause
