@echo off
cd /d "%~dp0"
title DeltaZero - Re-simulate replays (batch mode)
rem Start_DeltaZero.bat must not run at the same time (one Python connection per game).
rem Maps: the bulk list (data\bulk) if it exists, else the TMX map pool (data\pool, made here
rem on the first run: the 2000 most awarded maps of 5-60 s). Maps and replays are fetched as it
rem goes; replays already re-simulated are skipped, so a stopped run continues.
rem Uses every running helper instance.
set HOURS=
set /p HOURS=How many hours (empty = 3): 
if "%HOURS%"=="" set HOURS=3
if not exist data\bulk\maps.jsonl if not exist data\pool\maps.json python tmdriver.py tmx-pool --maps 2000
python tmdriver.py resim --replays 5 --hours %HOURS% --map 0
pause
