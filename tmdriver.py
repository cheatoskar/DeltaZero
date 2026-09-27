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
import time
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
                 'ghost-ft': runs / 'ghost_ft' / 'best.pt', 'ghost-big': runs / 'ghost_big' / 'best.pt',
                 'ghost-big2': runs / 'ghost_big2' / 'best.pt'}
        ckpt = named.get(args.model, Path(args.model))
        if not ckpt.exists():
            sys.exit(f'model not found: {ckpt}')
        os.environ['TMDRIVER_CKPT'] = str(ckpt)
        print(f'model: {ckpt}')
    if args.line:
        os.environ['TMDRIVER_LINE'] = '1'     # 'AI drive (live)' uses the reference line
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
    import json
    from tmdriver.bulk import MAPS_LOG
    from tmdriver.paths import POOL
    source, ids = args.source, []
    if args.maps:
        source, ids = 'tmx', [int(x) for x in args.maps.replace(' ', '').split(',') if x]
    elif args.pool or (args.map == 0 and not MAPS_LOG.exists() and POOL.exists()):
        if not POOL.exists():
            sys.exit(f'no map pool yet ({POOL}): run `python tmdriver.py tmx-pool` first')
        source = 'tmx'
        ids = [m['track_id'] for m in json.loads(POOL.read_text(encoding='utf-8'))['maps']]
        print(f'map pool: {len(ids)} maps from {POOL}')
    elif args.map is not None:
        # the button: a TMX id, or 0 = the bulk list if downloaded, else the pool, else the open map
        source = 'bulk' if args.map == 0 and MAPS_LOG.exists() else 'tmx'
        ids = [args.map]
    link = _link(args)
    from tmdriver.instances import connect_helpers
    helpers = connect_helpers(args.port) if args.helpers == 'auto' else []
    run_resim(link, only_missing=not args.all, max_replays=args.replays, limit=args.limit,
              source=source, start=args.start, hours=args.hours, batch=not args.per_tick,
              track_ids=ids, helpers=helpers)


def rl(args):
    from tmdriver.rl import rl_train
    link = _link(args)
    from tmdriver.instances import connect_helpers
    helpers = connect_helpers(args.port) if args.helpers == 'auto' else []
    rl_train(link, track_id=args.map, iterations=args.iterations, runs=args.runs, branch=args.branch,
             use_line=args.line, helpers=helpers, show=not args.no_show, ckpt=args.model or None, lr=args.lr,
             warmup=args.warmup, kl_coef=args.kl, seed=args.seed)


def check_hold(args):
    from tmdriver.improve import hold_check
    ok = hold_check(_link(args), track_id=args.map)
    if not ok:
        sys.exit('held runs differ: keep --hold off and send the lines above')


def night(args):
    from tmdriver.instances import connect_helpers
    from tmdriver.nightly import Night
    links = [_link(args)] + (connect_helpers(args.port) if args.helpers == 'auto' else [])
    ports = [l.sock.getpeername()[1] for l in links]
    Night(links, ports, hours=args.hours, n_maps=args.maps, replays=args.replays,
          batch=not args.per_tick).run()


def sim_night(args):
    os.environ['TMDRIVER_TMX_GAP'] = str(args.tmx_gap)
    from tmdriver import tmx
    tmx.MIN_GAP_S = args.tmx_gap
    from tmdriver.sim_night import SimNight
    SimNight(hours=args.hours, min_awards=args.min_awards, replays=args.replays,
             nadeo_replays=args.nadeo_replays, within=args.within, workers=args.workers).run()


def virtual(args):
    """Start virtual game instances (TMNF-C physics, no client) on ports port .. port+n-1 and keep
    them running until Ctrl+C. Tools then use --port <first port>; the rest are helpers."""
    import subprocess
    procs = []
    for k in range(args.instances):
        cmd = [sys.executable, '-u', '-m', 'tmdriver.virtual_game', '--port', str(args.port + k)]
        if args.map_file:
            cmd += ['--map', args.map_file]
        procs.append(subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent / 'src')))
    print(f'{args.instances} virtual game instance(s) on ports {args.port}..{args.port + args.instances - 1}; '
          f'use --port {args.port}. Ctrl+C stops them.')
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    for p in procs:
        p.terminate()


def trainpack(args):
    from tmdriver.trainpack import build
    build(zip_it=not args.no_zip)


def route_learn(args):
    from tmdriver.route_learn import run
    run(n_maps=args.maps, n_tune=args.tune, n_eval=args.eval, trials=args.trials)


def rl_vec(args):
    from tmdriver.rl_vec import rl_vec_train
    from tmdriver.virtual_game import map_dirs
    f = Path(args.map_file) if args.map_file else None
    if f is None:
        for d in map_dirs():
            hits = sorted(Path(d).glob(f'*_{args.map}.Challenge.Gbx'))
            if hits:
                f = hits[0]
                break
    if f is None:
        sys.exit(f'no map file for {args.map} (download it first, e.g. with sim-night or rl --map)')
    rl_vec_train(f, track_id=args.map, iterations=args.iterations, runs=args.runs, cars=args.cars,
                 use_line=args.line, ckpt=args.model or None, lr=args.lr, warmup=args.warmup, kl_coef=args.kl,
                 eval_temp=args.eval_temp, seed=args.seed)


def tmx_pool(args):
    import json
    from tmdriver import tmx
    from tmdriver.collect import holdout
    from tmdriver.paths import POOL
    maps = tmx.search_tracks(args.maps, int(args.min * 1000), int(args.max * 1000))
    for m in maps:
        m['holdout'] = holdout(m['track_id'])      # never trained on (evaluation), same hash as collect
    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps({'made': time.strftime('%Y-%m-%d %H:%M'), 'min_s': args.min, 'max_s': args.max,
                                'maps': maps}, indent=1), encoding='utf-8')
    print(f"{len(maps)} maps ({sum(m['holdout'] for m in maps)} held out) -> {POOL}")


def launch(args):
    from tmdriver.instances import launch_helpers
    link = _link(args)

    def status(text):
        link.status(text)
        link.flush()

    launch_helpers(args.helpers, base=args.port, status=status)


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
        orient_drop=args.orient_drop, out=args.out, seed=args.seed)


def improve(args):
    from tmdriver.improve import improve as run
    link = _link(args)
    from tmdriver.instances import connect_helpers
    helpers = connect_helpers(args.port) if args.helpers == 'auto' else []
    run(link, track_id=args.map, rounds=args.rounds, episodes=args.episodes, show=not args.no_show,
        ckpt=args.model or None, branch=args.branch, seed=args.seed, use_line=args.line, helpers=helpers)


def show(args):
    from tmdriver.improve import show_best
    show_best(_link(args), track_id=args.map, speed=args.speed)


def drive(args):
    from tmdriver.improve import drive_preview
    drive_preview(_link(args), track_id=args.map, speed=args.speed, ckpt=args.model or None, plans=args.plans,
                  use_line=args.line)


def ghost_replays(args):
    from tmdriver.replaybuild import build_all
    build_all(workers=args.workers, rebuild=args.rebuild, source=args.source)


GAME_COMMANDS = {'resim', 'improve', 'drive', 'show', 'eval', 'launch', 'rl', 'check-hold', 'night'}


def serve_running() -> bool:
    """Another `tmdriver.py serve` process on this PC (Windows; best effort, False if unknown).
    Seen 2026-09-25: a running serve reconnected and took the connection from a resim run."""
    if os.name != 'nt':
        return False
    import subprocess
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command',
                              "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                              "ForEach-Object { $_.CommandLine }"],
                             capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return False
    return any('tmdriver.py' in line and ' serve' in line for line in out.splitlines())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', default='auto', help='auto (CUDA if available) | cuda | cpu, for the live driver')
    ap.add_argument('--hold', type=int, default=0,
                    help='hold each AI action N ticks in the plugin (5 = one Python round trip per decision; '
                         'not yet verified in the real game)')
    ap.add_argument('--no-draw', action='store_true',
                    help='turn off rendering in the game(s) while this command runs (TMInterface draw_game)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('install-plugin').set_defaults(fn=install_plugin)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=P.PORT)
    s.add_argument('--model', default='', help="ghost | ghost-latest (pretrained) | best | last (M1) | path")
    s.add_argument('--line', action='store_true', help="'AI drive (live)' with the reference line (default: blocks only)")
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
    r.add_argument('--map', type=int, default=None,
                   help='one TMX map (fetched if needed); 0 = the bulk list if downloaded, else the open map')
    r.add_argument('--maps', default='', help='comma-separated TMX ids, done one after the other (fetched if needed)')
    r.add_argument('--pool', action='store_true', help='every map of the TMX map pool (tmx-pool), fetched as it goes')
    r.add_argument('--helpers', default='auto', help='auto: also use every running helper instance | 0: main only')
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
    pt.add_argument('--seed', type=int, default=0, help='data order (use another one when continuing with --init)')
    pt.set_defaults(fn=pretrain)
    im = sub.add_parser('improve', help='the model practises one map and gets faster (serve must NOT run)')
    im.add_argument('--map', type=int, default=None, help='TMX id (default: the map open in the game)')
    im.add_argument('--rounds', type=int, default=20)
    im.add_argument('--episodes', type=int, default=6, help='sampled runs per round (+1 greedy)')
    im.add_argument('--model', default='', help='checkpoint (default: as serve)')
    im.add_argument('--branch', type=int, default=8,
                    help='runs per round that start shortly before where the best run got stuck (0 = off)')
    im.add_argument('--seed', type=int, default=None, help='random seed (default: a new one each time)')
    im.add_argument('--no-show', action='store_true', help='do not show new best runs in the game')
    im.add_argument('--helpers', default='auto', help='auto: spread the runs over every running helper instance | 0: main only')
    im.add_argument('--line', action='store_true', help='give the model the reference line (fastest TMX replay; fetched if missing). Default: blocks only')
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
    dv.add_argument('--plans', type=int, default=12, help='planned runs (1 greedy + sampled); the best is driven')
    dv.add_argument('--model', default='')
    dv.add_argument('--line', action='store_true', help='give the model the reference line (fastest TMX replay; fetched if missing). Default: blocks only')
    dv.add_argument('--port', type=int, default=P.PORT)
    dv.set_defaults(fn=drive)
    rp = sub.add_parser('rl', help='RL v2 (PPO) on one map, over every running instance (serve must NOT run)')
    rp.add_argument('--map', type=int, default=None, help='TMX id (default: the map open in the game)')
    rp.add_argument('--iterations', type=int, default=30)
    rp.add_argument('--runs', type=int, default=8, help='sampled runs per iteration (+1 greedy, + branch runs)')
    rp.add_argument('--branch', type=int, default=4, help='runs that start before where the best run got stuck')
    rp.add_argument('--line', action='store_true', help='give the model the reference line (default: blocks only)')
    rp.add_argument('--lr', type=float, default=1e-5)
    rp.add_argument('--warmup', type=int, default=2, help='iterations that train only the critic')
    rp.add_argument('--kl', type=float, default=0.1, help='KL leash to the starting model')
    rp.add_argument('--model', default='', help='checkpoint (default: as serve)')
    rp.add_argument('--seed', type=int, default=None)
    rp.add_argument('--no-show', action='store_true')
    rp.add_argument('--helpers', default='auto')
    rp.add_argument('--port', type=int, default=P.PORT)
    rp.set_defaults(fn=rl)
    ch = sub.add_parser('check-hold', help='real-game check that --hold 5 gives the same runs (serve must NOT run)')
    ch.add_argument('--map', type=int, default=None, help='TMX id (default: the map open in the game)')
    ch.add_argument('--port', type=int, default=P.PORT)
    ch.set_defaults(fn=check_hold)
    ni = sub.add_parser('night', help='overnight: fetch top TMX maps + replays while re-simulating (serve must NOT run)')
    ni.add_argument('--hours', type=float, default=8.0)
    ni.add_argument('--maps', type=int, default=5000, help='size of the map pool (most awarded first)')
    ni.add_argument('--replays', type=int, default=5, help='fastest replays per map')
    ni.add_argument('--per-tick', action='store_true', help='answer every tick from Python (slower)')
    ni.add_argument('--helpers', default='auto')
    ni.add_argument('--port', type=int, default=P.PORT)
    ni.set_defaults(fn=night)
    tp = sub.add_parser('trainpack', help="sim-night's exact runs + maps -> one compact upload for a training server")
    tp.add_argument('--no-zip', action='store_true')
    tp.set_defaults(fn=trainpack)
    rl2 = sub.add_parser('route-learn', help='learn the route planner costs from record lines; held-out report')
    rl2.add_argument('--maps', type=int, default=400)
    rl2.add_argument('--tune', type=int, default=60)
    rl2.add_argument('--eval', type=int, default=150)
    rl2.add_argument('--trials', type=int, default=24)
    rl2.set_defaults(fn=route_learn)
    rv = sub.add_parser('rl-vec', help='RL v2 without the game, batched: N cars in one TMNF-C world')
    rv.add_argument('--map', type=int, default=None, help='TMX id')
    rv.add_argument('--map-file', default='')
    rv.add_argument('--iterations', type=int, default=50)
    rv.add_argument('--runs', type=int, default=64, help='sampled runs per iteration (+1 greedy)')
    rv.add_argument('--cars', type=int, default=32, help='cars driving at once')
    rv.add_argument('--line', action='store_true')
    rv.add_argument('--lr', type=float, default=1e-5)
    rv.add_argument('--warmup', type=int, default=2)
    rv.add_argument('--kl', type=float, default=0.1)
    rv.add_argument('--model', default='')
    rv.add_argument('--seed', type=int, default=None)
    rv.add_argument('--eval-temp', type=float, default=0.7,
                    help='temperature of the per-iteration evaluation run (argmax never finishes, see rl_vec.py)')
    rv.set_defaults(fn=rl_vec)
    vg = sub.add_parser('virtual', help='virtual game instances (TMNF-C physics, no game client) for rl/improve/drive')
    vg.add_argument('--instances', type=int, default=4)
    vg.add_argument('--port', type=int, default=8600)
    vg.add_argument('--map-file', default='', help='map to start on (else the first `map` command)')
    vg.set_defaults(fn=virtual)
    sn = sub.add_parser('sim-night', help='overnight WITHOUT the game: fetch TMX maps + replays, re-simulate in TMNF-C')
    sn.add_argument('--hours', type=float, default=8.0)
    sn.add_argument('--min-awards', type=int, default=2, help='maps with at least this many awards')
    sn.add_argument('--replays', type=int, default=20, help='at most this many replays per map')
    sn.add_argument('--within', type=float, default=1.05, help='only replays within this factor of the best time')
    sn.add_argument('--nadeo-replays', type=int, default=100, help="fastest replays on Nadeo's own maps")
    sn.add_argument('--workers', type=int, default=0, help='simulation processes (default: CPU cores - 1)')
    sn.add_argument('--tmx-gap', type=float, default=0.1, help='seconds between TMX requests')
    sn.set_defaults(fn=sim_night)
    tp = sub.add_parser('tmx-pool', help='map pool from the TMX search: most awarded maps in a time range')
    tp.add_argument('--maps', type=int, default=2000)
    tp.add_argument('--min', type=float, default=5.0, help='minimum author time in seconds')
    tp.add_argument('--max', type=float, default=60.0, help='maximum author time in seconds')
    tp.set_defaults(fn=tmx_pool)
    la = sub.add_parser('launch', help='start more game instances (helpers) and wait until they listen')
    la.add_argument('--helpers', type=int, default=2)
    la.add_argument('--port', type=int, default=P.PORT)
    la.set_defaults(fn=launch)
    gr = sub.add_parser('ghost-replays', help='stage A2: TMX replays -> shards with orientation (no game needed)')
    gr.add_argument('--workers', type=int, default=4)
    gr.add_argument('--rebuild', action='store_true')
    gr.add_argument('--source', default='bulk', help="bulk (replay_shards) | resim_c | resim_npz | trainpack (sim-night's exact runs -> resim_shards)")
    gr.set_defaults(fn=ghost_replays)
    args = ap.parse_args()
    os.environ['TMDRIVER_DEVICE'] = args.device
    if args.hold > 1:
        os.environ['TMDRIVER_HOLD'] = str(args.hold)
    if args.no_draw:
        os.environ['TMDRIVER_DRAW_GAME'] = '0'
    done = os.environ.get('TMDRIVER_JOB_DONE_FILE')
    if not done and args.cmd in GAME_COMMANDS and serve_running():
        sys.exit('Start_DeltaZero.bat (serve) is running: it would take the game connection away from '
                 'this command. Close its window first, or use the buttons in the game instead.')
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
        print('\nDone.')
    except KeyboardInterrupt:
        print('\nStopped.')
    except BaseException:
        import traceback
        traceback.print_exc()
        print('\nERROR (see above). Please send the last lines.')
    finally:
        Path(done).touch()     # the server takes the game connection back now
    try:
        input('Press Enter to close this window.')
    except EOFError:
        pass


if __name__ == '__main__':
    main()
