@echo off
cd /d "%~dp0"
title DeltaZero - RL v2 (PPO) on one map
rem Start_DeltaZero.bat must not run at the same time (one Python connection per game).
rem Uses every running helper instance. Ctrl+C ends it, saves and shows the best run.
rem Results: runs\rl\<map uid>\ (progress.jsonl, model.pt with the critic, best_run.json).
set MAP=
set ITER=
set LINE=
set /p MAP=TMX id of the map (empty = the map open in the game): 
set /p ITER=Iterations (empty = 30): 
set /p LINE=Use the reference line? y/N: 
if "%ITER%"=="" set ITER=30
set OPTS=--iterations %ITER%
if not "%MAP%"=="" set OPTS=%OPTS% --map %MAP%
if /i "%LINE%"=="y" set OPTS=%OPTS% --line
python tmdriver.py rl %OPTS%
pause
