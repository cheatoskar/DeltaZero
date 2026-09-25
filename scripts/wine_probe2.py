"""Server check: tick rate in simulation-only mode (the re-simulation path).

    python scripts/wine_probe2.py [ticks]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tmdriver import protocol as P  # noqa: E402
from tmdriver.link import Link  # noqa: E402
from tmdriver.session import Episode, GameSession  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


class Gas(Episode):
    def begin(self, start):
        self.n, self.t0 = 0, time.perf_counter()

    def act(self, st):
        self.n += 1
        if self.n >= N or st.finished:
            self.dt = time.perf_counter() - self.t0
            self.last = st.race_time
            return None
        return 0, 0, P.UP

    def result(self):
        return {'ticks': self.n, 'wall_s': round(self.dt, 2), 'rate': round(self.n / self.dt), 'race_ms': self.last}


link = Link.connect(port=P.PORT, wait_s=10)
sess = GameSession(link, print)
t0 = time.perf_counter()
res = sess.run([Gas(), Gas()], sim_only=True)
print('sim-only:', res, f'total {time.perf_counter() - t0:.1f}s incl. restart')
link.close()
