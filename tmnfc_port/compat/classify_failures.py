"""Why is a map not exact in TMNF-C? Dump its track from the game and re-run the replays on it.

    python compat/classify_failures.py N     (N maps that came out 0/x exact in data/resim_c)

exact on the in-game track  -> the offline track builder is wrong for this map
still not exact             -> physics (or start) problem in TMNF-C
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DZ = Path.home() / 'Downloads' / 'DeltaZero'
sys.path.insert(0, str(DZ / 'src'))
sys.path.insert(0, str(HERE))

from capture_tracks import capture  # noqa: E402
from tmdriver import replay as R, tmnfc  # noqa: E402
from tmdriver.paths import TMX, safe  # noqa: E402
from tmdriver.resim import table_action  # noqa: E402

OUT = HERE.parent / 'out' / 'classify.jsonl'


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rows = [json.loads(l) for l in (DZ / 'data/resim_c/index.jsonl').read_text(encoding='utf-8').splitlines()]
    done = set()
    if OUT.exists():
        done = {json.loads(l)['track_id'] for l in OUT.read_text(encoding='utf-8').splitlines()}
    by = {}
    for r in rows:
        if r.get('skipped'):
            continue
        by.setdefault(r['track_id'], []).append(r)
    maps = [(tid, rs) for tid, rs in by.items() if not any(r['exact'] for r in rs) and tid not in done][:n]
    for tid, rs in maps:
        ch = next((DZ / 'data/maps').glob(f'*_{tid}.Challenge.Gbx'))
        prep = tmnfc.prepare_map(ch, str(tid))
        game = HERE.parent / 'cap' / 'tracks' / f"{prep['sha'][:16]}.tmnftrack"
        row = {'track_id': tid, 'name': rs[0]['name'], 'offline_first_off_ms': rs[0].get('first_off_ms')}
        try:
            if not game.exists():
                capture(ch, game)
            ok = tot = 0
            first = []
            with tmnfc.MapSim(dict(prep, track=game)) as sim:
                for r in rs[:5]:
                    rep = R.load(TMX / safe(r['uid']) / f"{r['replay']}.Replay.Gbx")
                    tab = R.input_table(rep, steer_sign=-1)
                    n_t = (rep.race_time_ms + 200) // 10 + 1
                    c = tmnfc.ghost_check(sim.run([table_action(tab, k * 10, 1) for k in range(n_t)]), rep)
                    ok += c['exact']
                    tot += 1
                    first.append(c['first_off_ms'])
            row.update(ingame_exact=ok, ingame_total=tot, ingame_first_off_ms=first,
                       verdict='builder' if ok == tot else ('mixed' if ok else 'physics'))
        except Exception as e:                  # noqa: BLE001
            row['error'] = repr(e)[:200]
        print(json.dumps(row))
        sys.stdout.flush()
        with open(OUT, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
