@echo off
cd /d "%~dp0"
title DeltaZero - check that holding actions in the plugin gives identical runs
rem Open a map in the game first. Start_DeltaZero.bat must not run at the same time.
rem The last line must say: held runs identical to per-tick runs: True
python tmdriver.py check-hold
pause
