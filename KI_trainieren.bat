@echo off
cd /d "%~dp0"
title TMDriver - KI trainiert auf einer Map (improve)
set MAP=
set ROUNDS=
set /p MAP=TMX-ID der Map (leer = die Map, die gerade im Spiel laeuft): 
set /p ROUNDS=Runden (leer = 20): 
if "%ROUNDS%"=="" set ROUNDS=20
if "%MAP%"=="" (python tmdriver.py improve --rounds %ROUNDS%) else (python tmdriver.py improve --map %MAP% --rounds %ROUNDS%)
pause
