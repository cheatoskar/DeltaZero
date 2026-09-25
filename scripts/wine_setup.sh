#!/bin/bash
# Fresh Vast box (Ubuntu 24.04): TMNF + TMLoader + TMInterface under Wine, headless (Xvfb).
# Done by hand on 2026-09-25 (~1 h); this script repeats those steps (~15 min).
#
# 1) On the laptop (Git Bash), pack TMLoader WITHOUT personal data (no game profiles, no logins):
#      cd "$LOCALAPPDATA" && tar cf - TMLoader/TMLoader.exe TMLoader/sciter.dll TMLoader/ShimRun.exe \
#        TMLoader/config.yaml TMLoader/ui TMLoader/database/TmForever/profiles \
#        TMLoader/database/TmForever/products/TMInterface/2.2.1 \
#        TMLoader/database/TmForever/products/TMInterface/description.yaml \
#        TMLoader/database/TmForever/products/CoreMod TMLoader/database/TmForever/products/TmForever \
#        | xz -1 -T0 > /tmp/tmloader.tar.xz                                  # ~18 MB
#      scp -P PORT /tmp/tmloader.tar.xz root@HOST:/root/tmnf/
#    The TMDriverAI TMLoader profile is: program TmForever, mods TMInterface + CoreMod.
# 2) On the box: bash scripts/wine_setup.sh, then bash scripts/wine_start.sh.
#    The first start creates the in-game profile "root" (the click in wine_start.sh selects it).
#    Never copy cheatoskar's own TMNF profiles, pkey.dat or session.dat to rented servers.
set -e
mkdir -p /root/tmnf
cat > /root/tmnf/env.sh <<'EOF'
export DISPLAY=:99
export WINEPREFIX=/root/.wine
export WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml="
EOF
source /root/tmnf/env.sh

dpkg --add-architecture i386
apt-get update -q
apt-get install -y -q wine64 wine32:i386 xvfb xdotool cabextract winbind x11-utils imagemagick liblzo2-dev \
  >/root/tmnf/apt.log 2>&1
bash -lc 'pip install -q python-lzo'          # the .Gbx reader needs LZO on Linux

pgrep -x Xvfb >/dev/null || (nohup Xvfb :99 -screen 0 1280x800x24 -nolisten tcp >/root/tmnf/xvfb.log 2>&1 &)
sleep 2
wineboot -i >/root/tmnf/wineboot.log 2>&1

# TMNF, silently, from Nadeo's CDN
cd /root/tmnf
[ -f tmnationsforever_setup.exe ] || \
  curl -sL -o tmnationsforever_setup.exe https://nadeo-download.cdn.ubi.com/trackmaniaforever/tmnationsforever_setup.exe
timeout 900 wine tmnationsforever_setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /LANG=english \
  '/DIR=C:\Program Files (x86)\TmNationsForever' >/root/tmnf/install.log 2>&1 || true

# TMLoader (+ TMInterface 2.2.1, CoreMod) from the laptop tarball
L=/root/.wine/drive_c/users/root/AppData/Local
mkdir -p "$L" && cd "$L" && tar xJf /root/tmnf/tmloader.tar.xz

# Plugin + maps (repo checked out in ~/TMDriverAI)
P=/root/.wine/drive_c/users/root/Documents/TMInterface/Plugins/TMDriver
mkdir -p "$P" && cp ~/TMDriverAI/plugin/TMDriver/*.as "$P"/
cd ~/TMDriverAI && bash -lc 'python scripts/install_maps.py' || echo "(maps: run install_maps.py after data/maps exists)"
echo "setup done: now bash ~/TMDriverAI/scripts/wine_start.sh"
