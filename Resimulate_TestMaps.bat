@echo off
cd /d "%~dp0"
title DeltaZero - Re-simulation pipeline test (20 TMX maps x 5 replays)
rem 20 TMX maps checked on 2026-09-25 (parse, replays without respawns). Missing maps and replays
rem are fetched from TMX; replays already re-simulated are skipped, so a restart continues.
rem Uses every running helper instance. Start_DeltaZero.bat must NOT run at the same time.
python tmdriver.py resim --replays 5 --maps 10036840,10030774,414041,2481743,3706049,3706054,3706064,3707173,3707182,3709629,3709653,3748608,3905537,3937465,3937477,3937479,5240131,5490289,9288288,10043
pause
