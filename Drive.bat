@echo off
cd /d "%~dp0"
title DeltaZero - Drive (plan the line, draw it, drive it)
rem Start_DeltaZero.bat must not run at the same time (one Python connection per game).
set MAP=
set LINE=
set /p MAP=TMX id of the map (empty = the map open in the game): 
set /p LINE=Use the reference line? y/N: 
set OPTS=
if not "%MAP%"=="" set OPTS=%OPTS% --map %MAP%
if /i "%LINE%"=="y" set OPTS=%OPTS% --line
python tmdriver.py drive %OPTS%
pause
