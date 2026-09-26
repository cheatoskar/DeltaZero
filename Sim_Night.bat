@echo off
cd /d "%~dp0"
title DeltaZero - re-simulation WITHOUT the game (TMNF-C): download TMX maps + replays, simulate on all cores
rem No game needed. Maps with at least 2 awards, most awarded first; up to 20 replays per map
rem within 5 %% of the best time (100 on Nadeo maps). Stop with Ctrl+C; the next start continues.
rem Progress: data\resim_c\status.json   Results: data\resim_c\   Maps: data\maps\   Replays: data\tmx\
set HOURS=
set /p HOURS=How many hours (empty = 8):
if "%HOURS%"=="" set HOURS=8
python tmdriver.py sim-night --hours %HOURS%
pause
