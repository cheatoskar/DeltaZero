"""Server check: can Python reach the plugin in the Wine game and load a map by command?

    python scripts/wine_probe.py [TMX id]      (default 2233 = A01-Race)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tmdriver import protocol as P  # noqa: E402
from tmdriver.link import Link  # noqa: E402
from tmdriver.session import GameSession  # noqa: E402

tid = int(sys.argv[1]) if len(sys.argv) > 1 else 2233
m = next(json.loads(l) for l in (ROOT / 'data/bulk/maps.jsonl').read_text(encoding='utf-8').splitlines()
         if json.loads(l)['track_id'] == tid)
t0 = time.monotonic()
link = Link.connect(port=P.PORT, wait_s=20)
print(f'connected in {time.monotonic() - t0:.1f}s')
sess = GameSession(link, print)
t0 = time.monotonic()
ok = sess.load_map(m['map_file'], m['uid'])
print(f"load_map {m['name']!r}: {ok} after {time.monotonic() - t0:.1f}s")
if ok:
    # count steps for 3 s of race to measure the Python-in-the-loop rate
    link.mode(P.MODE_TEST)
    link.flush()
    n, t0, first, last = 0, time.monotonic(), None, None
    while time.monotonic() - t0 < 5:
        kind, st = sess.next(timeout=10)
        if kind == P.P_STEP:
            first = st.race_time if first is None else first
            last = st.race_time
            n += 1
            link.action(0, 0, P.UP)
    print(f'{n} steps in 5 s wall ({n / 5:.0f}/s), race time {first} -> {last}')
    link.mode(P.MODE_IDLE)
    link.action(0, 0, 0)
link.close()
