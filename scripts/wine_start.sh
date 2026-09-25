#!/bin/bash
# Server (Vast, Wine): start TMNF with TMInterface and click through to the main menu.
# Coordinates measured on Xvfb 1280x800; clicks need ~1 s presses (software rendering, low fps).
source ~/tmnf/env.sh
pkill -x TmForever.exe; pkill -x TMLoader.exe; sleep 3   # -x: -f would also match the calling shell
cd /root/.wine/drive_c/users/root/AppData/Local/TMLoader
nohup wine TMLoader.exe run TmForever "TMDriverAI" > ~/tmnf/game.log 2>&1 &
click() { xdotool mousemove --sync $1 $2; sleep 0.5; timeout 5 xdotool mousedown 1; sleep ${3:-1.2}; timeout 5 xdotool mouseup 1; }
sleep 50
# move TMI's Input Editor off the "Stay offline" button
xdotool mousemove --sync 700 249; sleep 0.3; timeout 5 xdotool mousedown 1; sleep 0.4
for y in 300 400 500 600 700 760 785; do timeout 5 xdotool mousemove --sync 700 $y; sleep 0.3; done
timeout 5 xdotool mouseup 1; sleep 2
click 640 460; sleep 8
click 641 461 1.5; sleep 15        # Stay offline (the first press often only highlights it)
click 680 262; sleep 20            # profile
rm -f ~/tmnf/start.png; timeout 30 import -window root ~/tmnf/start.png
pgrep -f TmForever >/dev/null && echo "game running" || echo "GAME NOT RUNNING"
