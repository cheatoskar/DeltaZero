"""TMDriverAI command line.

    python tmdriver.py install-plugin   copy plugin/TMDriver into Documents/TMInterface/Plugins
    python tmdriver.py serve            connect to the plugin and handle its buttons
    python tmdriver.py collect          M1: choose maps from MX metadata, download maps + top replays
    python tmdriver.py resim            M1: replay the TMX inputs in the game -> full 10 ms state
    python tmdriver.py build            M1: -> data/m1/dataset.npz
    python tmdriver.py train            M1: -> runs/m1/driver.pt  (CPU, --minutes)
    python tmdriver.py eval             M1: the model drives held-out maps in the game
    python tmdriver.py bulk-fetch       download top-awarded TMX maps + replays (resumable, --rate)
    python tmdriver.py ghost-build      stage A: HF traces + MX blocks -> data/ghost/shards
    python tmdriver.py pretrain         stage A: train the ghost-feature driver (GPU) -> runs/ghost
    python tmdriver.py drive            plan the line (drawn in the game), then drive it
    python tmdriver.py improve          the model practises the current map and gets faster (RL v1)
    python tmdriver.py ghost-replays    stage A2: TMX replays -> shards with orientation (no game)
    python tmdriver.py fetch-tmx ID...  download TMX maps (into TmForever/Tracks/Challenges/TMDriver)
                                        and their fastest replays (into data/tmx/<uid>/) for the self test
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from tmdriver import protocol as P  # noqa: E402


def plugin_dir() -> Path:
    return Path(os.path.expanduser('~')) / 'Documents' / 'TMInterface' / 'Plugins'


def install_plugin(args):
    src = Path(__file__).resolve().parent / 'plugin' / 'TMDriver'
    root = plugin_dir()
    if not root.parent.exists():
        sys.exit(f'{root.parent} does not exist yet. Start TMNF once with TMInterface '
                 f'(TMLoader), then run this again.')
    dst = root / 'TMDriver'
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob('*.as'):
        shutil.copy2(f, dst / f.name)
        print(f'copied {f.name} -> {dst}')
    print('TMInterface reloads changed plugins automatically; otherwise enable it in Settings > Plugins.')


def serve(args):
    if args.model:
        runs = Path(__file__).resolve().parent / 'runs'
        named = {'best': runs / 'm1' / 'driver.pt', 'last': runs / 'm1' / 'driver_last.pt',
                 'ghost': runs / 'ghost' / 'best.pt', 'ghost-latest': runs / 'ghost' / 'latest.pt',
                 'ghost-ft': runs / 'ghost_ft' / 'best.pt'}
        ckpt = named.get(args.model, Path(args.model))
        if not ckpt.exists():
            sys.exit(f'model not found: {ckpt}')
        os.environ['TMDRIVER_CKPT'] = str(ckpt)
        print(f'model: {ckpt}')
    from tmdriver.server import serve as run
    run(port=args.port)


def fetch_tmx(args):
    from tmdriver import replay, tmx
    from tmdriver.paths import TMX, safe
    for tid in args.ids:
        res = tmx.fetch(tid, args.replays, TMX, safe)
        for rp in res['replays']:
            rep = replay.load(rp)
            ok = 'OK' if rep.map_uid == res['uid'] else f'UID MISMATCH ({rep.map_uid})'
            kind = 'analog' if rep.uses_analog_steer else 'keyboard'
            print(f'    {rp.name}: {rep.race_time_ms / 1000:.2f}s {kind}, respawns {rep.respawns}, '
                  f'{rep.game_version}, map uid {ok}')


def collect(args):
    from tmdriver.collect import collect as run
    run(args.maps, args.replays, must=args.must)


def _link(args):
    from tmdriver.link import Link
    return Link.connect(port=args.port, wait_s=10)


def resim(args):
    from tmdriver.resim import run_resim
    run_resim(_link(args), only_missing=not args.all, max_replays=args.replays, limit=args.limit,
              source=args.source, start=args.start, hours=args.hours, batch=not args.per_tick)


def build(args):
    from tmdriver.dataset import build as run
    run(stride=args.stride)


def train(args):
    from tmdriver.train import train as run
    run(minutes=args.minutes, d=args.d, layers=args.layers)


def evaluate(args):
    from tmdriver.evaluate import run_eval
    run_eval(_link(args), which=args.maps, watch=args.watch, speed=args.speed, limit=args.limit)


def bulk_fetch(args):
    from tmdriver.bulk import run
    run(args.maps, n_replays=args.replays, rate=args.rate, hours=args.hours, threads_n=args.threads)


def ghost_build(args):
    from tmdriver.tracebuild import build_all
    build_all(Path(args.traces), Path(args.blocks), workers=args.workers, stride=args.stride)


def pretrain(args):
    from tmdriver.pretrain import pretrain as run
    run(hours=args.hours, bs=args.bs, lr=args.lr, d=args.d, layers=args.layers, heads=args.heads,
        workers=args.workers, max_parts=args.max_parts, init=args.init, compile_model=args.compile,
        eval_min=args.eval_min, val_n=args.val_n, shards=args.shards, orient=args.orient,
        orient_drop=args.orient_drop, out=args.out)


def improve(args):
    from tmdriver.improve import improve as run
    run(_link(args), track_id=args.map, rounds=args.rounds, episodes=args.episodes, show=not args.no_show,
        ckpt=args.model or None)


def show(args):
    from tmdriver.improve import show_best
    show_best(_link(args), track_id=args.map, speed=args.speed)


def drive(args):
    from tmdriver.improve import drive_preview
    drive_preview(_link(args), track_id=args.map, speed=args.speed, ckpt=args.model or None)


def ghost_replays(args):
    from tmdriver.replaybuild import build_all
    build_all(workers=args.workers, rebuild=args.rebuild)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='auto', help='auto (CUDA if available) | cuda | cpu, for the live driver')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('install-plugin').set_defaults(fn=install_plugin)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=P.PORT)
    s.add_argument('--model', default='', help="ghost | ghost-latest (pretrained) | best | last (M1) | path")
    s.set_defaults(fn=serve)
    f = sub.add_parser('fetch-tmx')
    f.add_argument('ids', type=int, nargs='+')
    f.add_argument('--replays', type=int, default=3)
    f.set_defaults(fn=fetch_tmx)
    c = sub.add_parser('collect', help='M1: pick maps from MX metadata, download maps + top replays')
    c.add_argument('--maps', type=int, default=80)
    c.add_argument('--replays', type=int, default=5)
    c.add_argument('--must', type=int, nargs='*', default=[10036840, 10030774, 10043])
    c.set_defaults(fn=collect)
    r = sub.add_parser('resim', help='M1: replay TMX inputs in the game (serve must NOT run)')
    r.add_argument('--replays', type=int, default=5)
    r.add_argument('--all', action='store_true', help='redo replays that were already simulated')
    r.add_argument('--limit', type=int, default=0, help='only the first N maps (trial run)')
    r.add_argument('--source', default='m1', help='m1 (80-map manifest) | bulk (data/bulk/maps.jsonl -> data/resim)')
    r.add_argument('--start', type=int, default=0, help='skip the first N maps (split work between games)')
    r.add_argument('--hours', type=float, default=0.0, help='stop after this long')
    r.add_argument('--per-tick', action='store_true',
                   help='answer every tick from Python (slower; default: the plugin plays each replay itself)')
    r.add_argument('--port', type=int, default=P.PORT)
    r.set_defaults(fn=resim)
    b = sub.add_parser('build', help='M1: re-simulated runs -> data/m1/dataset.npz')
    b.add_argument('--stride', type=int, default=3)
    b.set_defaults(fn=build)
    t = sub.add_parser('train', help='M1: behaviour cloning -> runs/m1/driver.pt')
    t.add_argument('--minutes', type=float, default=12.0)
    t.add_argument('--d', type=int, default=96)
    t.add_argument('--layers', type=int, default=2)
    t.set_defaults(fn=train)
    e = sub.add_parser('eval', help='M1: the model drives maps in the game (serve must NOT run)')
    e.add_argument('--maps', default='holdout', help='holdout | train | all | bulk-holdout | bulk | comma-separated TMX ids')
    e.add_argument('--limit', type=int, default=0)
    e.add_argument('--watch', action='store_true', help='render and drive at --speed instead of simulation-only')
    e.add_argument('--speed', type=float, default=1.0)
    e.add_argument('--port', type=int, default=P.PORT)
    e.set_defaults(fn=evaluate)
    bf = sub.add_parser('bulk-fetch', help='download top-awarded TMX maps + replays (resumable)')
    bf.add_argument('--maps', type=int, default=2000)
    bf.add_argument('--replays', type=int, default=5)
    bf.add_argument('--rate', type=float, default=1.0, help='requests per second')
    bf.add_argument('--hours', type=float, default=1.5)
    bf.add_argument('--threads', type=int, default=4)
    bf.set_defaults(fn=bulk_fetch)
    gb = sub.add_parser('ghost-build', help='stage A: HF traces + MX blocks -> training shards')
    gb.add_argument('--traces', default='data/hf/traces')
    gb.add_argument('--blocks', default='data/hf/blocks')
    gb.add_argument('--workers', type=int, default=4)
    gb.add_argument('--stride', type=int, default=4, help='keep every Nth 100 ms sample')
    gb.set_defaults(fn=ghost_build)
    pt = sub.add_parser('pretrain', help='stage A: behaviour cloning on the trace shards')
    pt.add_argument('--hours', type=float, default=1.0)
    pt.add_argument('--bs', type=int, default=4096)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--d', type=int, default=256)
    pt.add_argument('--layers', type=int, default=6)
    pt.add_argument('--heads', type=int, default=8)
    pt.add_argument('--workers', type=int, default=6)
    pt.add_argument('--max-parts', type=int, default=0)
    pt.add_argument('--init', default='')
    pt.add_argument('--compile', action='store_true')
    pt.add_argument('--eval-min', type=float, default=10.0)
    pt.add_argument('--val-n', type=int, default=200000)
    pt.add_argument('--shards', default='', help="'root:weight,...' e.g. data/ghost/replay_shards:0.7,data/ghost/shards:0.3")
    pt.add_argument('--orient', action='store_true', help='use the car orientation (replay data)')
    pt.add_argument('--orient-drop', type=float, default=0.3)
    pt.add_argument('--out', default='', help='checkpoint folder (default runs/ghost)')
    pt.set_defaults(fn=pretrain)
    im = sub.add_parser('improve', help='the model practises one map and gets faster (serve must NOT run)')
    im.add_argument('--map', type=int, default=None, help='TMX id (default: the map open in the game)')
    im.add_argument('--rounds', type=int, default=20)
    im.add_argument('--episodes', type=int, default=6, help='sampled runs per round (+1 greedy)')
    im.add_argument('--model', default='', help='checkpoint (default: as serve)')
    im.add_argument('--no-show', action='store_true', help='do not show new best runs in the game')
    im.add_argument('--port', type=int, default=P.PORT)
    im.set_defaults(fn=improve)
    sh = sub.add_parser('show', help='play the best run that improve found on this map')
    sh.add_argument('--map', type=int, default=None, help='TMX id (default: the map open in the game)')
    sh.add_argument('--speed', type=float, default=1.0)
    sh.add_argument('--port', type=int, default=P.PORT)
    sh.set_defaults(fn=show)
    dv = sub.add_parser('drive', help='plan the line (drawn in the game), then drive it (serve must NOT run)')
    dv.add_argument('--map', type=int, default=None, help='TMX id (default: the map open in the game)')
    dv.add_argument('--speed', type=float, default=1.0)
    dv.add_argument('--model', default='')
    dv.add_argument('--port', type=int, default=P.PORT)
    dv.set_defaults(fn=drive)
    gr = sub.add_parser('ghost-replays', help='stage A2: TMX replays -> shards with orientation (no game needed)')
    gr.add_argument('--workers', type=int, default=4)
    gr.add_argument('--rebuild', action='store_true')
    gr.set_defaults(fn=ghost_replays)
    args = ap.parse_args()
    os.environ['TMDRIVER_DEVICE'] = args.device
    done = os.environ.get('TMDRIVER_JOB_DONE_FILE')
    if not done:
        args.fn(args)
        return
    # Started by a tool button in the game (serve): own console, keep it open at the end.
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(os.environ.get('TMDRIVER_JOB_TITLE', 'TMDriver'))
        except Exception:
            pass
    try:
        args.fn(args)
        print('\nFertig.')
    except KeyboardInterrupt:
        print('\nGestoppt.')
    except BaseException:
        import traceback
        traceback.print_exc()
        print('\nFEHLER (siehe oben). Bitte die letzten Zeilen schicken.')
    finally:
        Path(done).touch()     # the server takes the game connection back now
    try:
        input('Enter schliesst dieses Fenster.')
    except EOFError:
        pass


if __name__ == '__main__':
    main()
