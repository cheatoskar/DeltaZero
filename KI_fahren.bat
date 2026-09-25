@echo off
cd /d "%~dp0"
title TMDriver - KI faehrt (Linie planen, einzeichnen, fahren)
set MAP=
set /p MAP=TMX-ID der Map (leer = die Map, die gerade im Spiel laeuft): 
if "%MAP%"=="" (python tmdriver.py drive) else (python tmdriver.py drive --map %MAP%)
pause
