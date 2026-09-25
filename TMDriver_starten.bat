@echo off
cd /d "%~dp0"
title TMDriver - Server (dieses Fenster offen lassen)
echo TMDriver-Server: verbindet sich mit dem Spiel. Alles Weitere ueber das TMDriver-Fenster im Spiel.
echo Dieses Fenster offen lassen; schliessen beendet die Verbindung.
echo.
python tmdriver.py serve
pause
