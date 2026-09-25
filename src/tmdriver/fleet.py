"""Episodes on several game instances at once (the main one plus helpers, see instances.py).

A Fleet runs one group of episodes per game session, each in its own thread, and returns the
results group by group. Episodes that depend on each other (a PrefixEpisode saves a state in
its game, the BranchEpisodes after it start there) must be in the same group.

Every session drives with its own copy of the policy (policy_view): the per-run state
(history, heading, progress) is separate, the model and the map tensors are shared, so a
fine-tune after the round changes the driver of every instance.

Ctrl+C while a round runs sets every session's stop event: each ends its remaining episodes
at the next tick, the threads finish, and the KeyboardInterrupt is raised again for the
caller (which then calls recover()).
"""
import contextlib
import copy
import threading
from typing import List

from .session import Episode, GameSession


def policy_view(policy):
    """A second driver that shares the model with `policy` but has its own run state."""
    view = copy.copy(policy)
    view.restart()
    return view


class Fleet:
    def __init__(self, main: GameSession, helpers: List[GameSession] = ()):
        self.sessions = [main] + list(helpers)
        self.stop = threading.Event()
        for s in self.sessions:
            s.stop = self.stop

    @property
    def main(self) -> GameSession:
        return self.sessions[0]

    def __len__(self):
        return len(self.sessions)

    def run(self, groups: List[List[Episode]], sim_only: bool = True, keep: bool = False) -> List[List[dict]]:
        """groups[i] runs on sessions[i] (missing or empty groups: that session idles)."""
        groups = list(groups) + [[] for _ in range(len(self.sessions) - len(groups))]
        results: List[List[dict]] = [[] for _ in self.sessions]
        errors = []
        waiting = []                         # hold() contexts of instances that finished first
        lock = threading.Lock()

        def work(i):
            s = self.sessions[i]
            try:
                results[i] = s.run(groups[i], sim_only=sim_only, keep=keep)
                if keep:
                    # This instance is frozen in a STEP now; the others may need much longer than
                    # the plugin's 5 s answer timeout (seen 2026-09-25: the helper dropped while
                    # the main instance still drove its greedy run). Keep it waiting until all are done.
                    cm = s.hold()
                    cm.__enter__()
                    with lock:
                        waiting.append(cm)
            except BaseException as e:       # reported to the caller after all threads ended
                errors.append((i, e))
                self.stop.set()

        busy = [i for i, g in enumerate(groups) if g]
        for i, s in enumerate(self.sessions):
            if i not in busy:                # idle this round but maybe frozen from the last one
                cm = s.hold()
                cm.__enter__()
                waiting.append(cm)
        try:
            if len(busy) == 1:
                work(busy[0])               # no thread for a single instance (plain Ctrl+C)
            else:
                threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in busy]
                for t in threads:
                    t.start()
                try:
                    for t in threads:
                        while t.is_alive():
                            t.join(0.2)
                except KeyboardInterrupt:
                    self.stop.set()
                    for t in threads:
                        t.join(30.0)
                    raise
        finally:
            for cm in waiting:              # the pings stop; the caller holds again if it needs to
                cm.__exit__(None, None, None)
        if errors:
            self.stop.set()
            raise errors[0][1]
        if self.stop.is_set():
            raise KeyboardInterrupt('stopped')
        return results

    @contextlib.contextmanager
    def hold(self, skip_main: bool = False):
        """Keep every waiting instance frozen while Python works (GameSession.hold for each)."""
        with contextlib.ExitStack() as stack:
            for k, s in enumerate(self.sessions):
                if skip_main and k == 0:
                    continue
                stack.enter_context(s.hold())
            yield

    def release(self):
        for s in self.sessions:
            s.release()

    def recover(self):
        """After Ctrl+C / Stop: every instance back to normal (see GameSession.recover)."""
        self.stop.clear()
        for s in self.sessions:
            s.recover()
