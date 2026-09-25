"""Main loop: read plugin messages, hand steps to the task of the current mode."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import protocol as P
from .calib import Calibration
from .link import Link
from .paths import CALIBRATION, LIVE_MAPS, safe
from .tasks import Context, Driver, Recorder, SelfTest, Task

TASKS = {P.MODE_RECORD: Recorder, P.MODE_DRIVE: Driver, P.MODE_TEST: SelfTest}
ROOT = Path(__file__).resolve().parents[2]
JOB_NAMES = {P.JOB_DRIVE: 'Drive', P.JOB_TRAIN: 'Train', P.JOB_RESIM: 'Re-simulate replays', P.JOB_SHOW: 'Show best run'}


class JobRequest(Exception):
    """A tool button: the server hands the game connection to a job in its own console."""

    def __init__(self, job, track_id, rounds, minutes, flags):
        super().__init__(JOB_NAMES.get(job, job))
        self.job, self.track_id, self.rounds, self.flags = job, track_id, rounds, flags
        self.hours = round(max(minutes, 6) / 60.0, 2)

    def command(self):
        cmd = [sys.executable, '-u', str(ROOT / 'tmdriver.py'),
               '--device', 'auto' if self.flags & P.JOB_GPU else 'cpu']
        m = ['--map', str(self.track_id)] if self.track_id > 0 else []
        if self.job == P.JOB_DRIVE:
            return cmd + ['drive'] + m
        if self.job == P.JOB_TRAIN:
            return cmd + ['improve', '--rounds', str(max(1, self.rounds))] + m
        if self.job == P.JOB_RESIM:
            return cmd + ['resim', '--source', 'bulk', '--replays', '5', '--hours', str(self.hours)] + \
                (['--per-tick'] if self.flags & P.JOB_PER_TICK else [])
        if self.job == P.JOB_SHOW:
            return cmd + ['show'] + m
        raise ValueError(f'unknown job {self.job}')


def spawn_job(req: JobRequest, done_file: Path):
    """Start the job in a new console window (Windows) with live output. It touches
    done_file when it has finished (its window stays open until Enter)."""
    env = dict(os.environ, TMDRIVER_JOB_DONE_FILE=str(done_file), TMDRIVER_JOB_TITLE=f'TMDriver - {req}')
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    return subprocess.Popen(req.command(), cwd=str(ROOT), env=env, creationflags=flags)


def run_job(req: JobRequest, spawn=spawn_job, log=print):
    done = ROOT / 'runs' / 'job_done.flag'
    done.parent.mkdir(parents=True, exist_ok=True)
    done.unlink(missing_ok=True)
    log(f'job: {req} -> {" ".join(req.command()[2:])}')
    proc = spawn(req, done)
    while not done.exists() and proc.poll() is None:
        time.sleep(0.5)
    done.unlink(missing_ok=True)
    log(f'job {req} finished; reconnecting to the game')


class Server:
    def __init__(self, link: Link, log=print, calibration_path=CALIBRATION):
        self.link = link
        self.log = log
        self.ctx = Context(Calibration(calibration_path), log)
        self.task: Optional[Task] = None
        self.ctx.policy()   # torch import + weights now, not when the first STEP waits

    def handle(self, kind: int, payload) -> None:
        link = self.link
        if kind == P.P_HELLO:
            (version,) = payload
            if version != P.PROTOCOL:
                raise RuntimeError(f'plugin protocol {version}, python expects {P.PROTOCOL}: '
                                   f'copy the current plugin into TMInterface/Plugins')
            self.log('plugin connected')
            link.status('Python connected. Pick a map, then Drive or Train.')
            link.flush()

        elif kind == P.P_MAP:
            self.ctx.map = payload
            LIVE_MAPS.mkdir(parents=True, exist_ok=True)
            (LIVE_MAPS / f'{safe(payload.uid)}.json').write_text(json.dumps({
                'uid': payload.uid, 'name': payload.name, 'author': payload.author,
                'blocks': [b.__dict__ for b in payload.blocks]}))
            self.log(f'map: {payload.name!r} ({payload.uid}), {len(payload.blocks)} blocks')

        elif kind == P.P_UI:
            mode, speed = payload
            self.ctx.ui_speed = max(1.0, float(speed))
            self.log(f'button: {P.MODE_NAMES.get(mode, mode)} (speed {self.ctx.ui_speed:.1f})')
            cls = TASKS.get(mode)
            if mode == P.MODE_DRIVE:
                self.ctx.policy()   # load before the race starts: steps must be answered fast
            self.task = cls(link, self.ctx) if cls else None
            if self.task is None:
                link.status('Stopped.')
                link.flush()

        elif kind == P.P_JOB:
            raise JobRequest(*payload)

        elif kind == P.P_BENCH:
            if self.task is not None:
                self.task.on_bench(*payload)

        elif kind == P.P_STEP:
            task = self.task
            if task is None or task.done:
                # The plugin is streaming or waiting although no task runs (e.g. Python was
                # restarted mid-drive). Put it back to idle; the ACTION releases a waiting
                # plugin and is ignored otherwise.
                if task is None or task.blocking:
                    link.mode(P.MODE_IDLE)
                    link.action(0, 0, 0)
                self.task = None
                return
            task.on_step(payload)
            if task.done:
                self.task = None

    def run(self):
        while True:
            kind, payload = self.link.read_message()
            self.handle(kind, payload)


def serve(host: str = P.HOST, port: int = P.PORT, wait_s: float = 3600.0, spawn=spawn_job,
          max_jobs: int = 0):
    """max_jobs > 0 returns after that many jobs (tests)."""
    jobs = 0
    while True:
        link = Link.connect(host, port, wait_s=wait_s)
        job = None
        try:
            Server(link).run()
        except JobRequest as j:
            job = j
            link.status(f'{j}: running in its own console window.')
            link.flush()
        except ConnectionError as e:
            print(f'connection lost: {e}; reconnecting')
        finally:
            link.close()
        if job is not None:
            run_job(job, spawn)
            jobs += 1
            if max_jobs and jobs >= max_jobs:
                return
        time.sleep(1.0)
