@echo off
cd /d "%~dp0"
title DeltaZero - server (keep this window open)
echo DeltaZero server: connects to the game. Everything else happens in the DeltaZero window in the game.
echo Keep this window open; closing it ends the connection.
echo.
python tmdriver.py serve
pause
