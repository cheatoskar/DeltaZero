@echo off
cd /d "%~dp0"
title TMDriver - Nachsimulation (Replays im Spiel abspielen, Batchmodus)
set HOURS=
set /p HOURS=Wie viele Stunden (leer = 3): 
if "%HOURS%"=="" set HOURS=3
python tmdriver.py resim --source bulk --replays 5 --hours %HOURS%
pause
