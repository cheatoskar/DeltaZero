@echo off
cd /d "%~dp0"
title DeltaZero - Train on one map (improve)
rem Start_DeltaZero.bat must not run at the same time (one Python connection per game).
rem Ctrl+C in this window ends the training, saves the best run and shows it in the game.
set MAP=
set ROUNDS=
set LINE=
set /p MAP=TMX id of the map (empty = the map open in the game): 
set /p ROUNDS=Rounds (empty = 20): 
set /p LINE=Use the reference line? y/N: 
if "%ROUNDS%"=="" set ROUNDS=20
set OPTS=--rounds %ROUNDS%
if not "%MAP%"=="" set OPTS=%OPTS% --map %MAP%
if /i "%LINE%"=="y" set OPTS=%OPTS% --line
python tmdriver.py improve %OPTS%
pause
