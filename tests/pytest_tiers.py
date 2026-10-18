"""Test tiers from *measured* durations, so a development tier cannot silently grow.

This file is vendored, byte-identical, into every test suite of the family as
``tests/pytest_tiers.py`` and loaded from ``tests/conftest.py`` with
``pytest_plugins = ("pytest_tiers",)``. Edit the canonical copy and re-sync every
suite; a suite's copy is not the place to change it.

Why it exists
-------------
A ``slow`` marker is a judgement made once, per test, by whoever wrote the test. As a
suite grows, the unmarked part grows with it, and nothing reports when the "quick"
selection has itself become an hour-long run. This plugin replaces the judgement with
a measurement and reports the drift every time the quick tier runs.

What it does
------------
``--tier quick``
    Deselects tests marked ``slow`` **and** tests whose recorded duration exceeds
    ``--tier-max`` seconds (ini ``tier_max_seconds``, default 60). A test with no
    recorded duration is *kept*: a new test is run until it has been measured.
    A test marked ``essential`` is kept regardless -- use it for a gate that must run
    in every tier however much it costs.

``--tier all`` (the default)
    Changes nothing. A bare ``pytest`` still runs every test.

``--record-durations``
    After the run, merges each test's setup + call + teardown time into
    ``tests/durations.json`` (next to this file) with the date, and appends the run's
    provenance (commit, host, load average, xdist workers). It is rewritten every
    60 s during the run, so an interrupted recording keeps what it measured (its
    provenance says ``"complete": false``). Commit the file: it is the measurement
    the quick tier is built from.

Every run ends with a one-line tier summary. A quick-tier run whose wall time
exceeds ``tier_budget_seconds`` (default 600) says so in a warning, which is the
signal to re-record durations or to look at what the tier now contains.

What it cannot see
------------------
* Module- or session-scoped fixture cost is charged to the first test that uses the
  fixture. Deselecting that test moves the cost to the next one, so the estimate
  printed before a quick run can be low; the wall time printed after it is not.
* Durations are from one machine at one load. Under ``-n`` they include contention.
  They are an ordering, not a benchmark -- never quote them as performance numbers.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from datetime import date
from pathlib import Path

import pytest

DURATIONS_FILE = Path(__file__).with_name("durations.json")
_KEEP_RUNS = 10            # provenance entries kept in durations.json
_FLUSH_EVERY_S = 60.0      # --record-durations rewrites the file this often


def pytest_addoption(parser):
    group = parser.getgroup("tiers", "test tiers from recorded durations")
    group.addoption(
        "--tier", choices=("all", "quick"), default="all",
        help="quick: drop `slow` tests and tests recorded slower than --tier-max "
             "(unless marked `essential`). Default: all.")
    group.addoption(
        "--tier-max", type=float, default=None, metavar="SECONDS",
        help="duration threshold for --tier quick (default: ini tier_max_seconds).")
    group.addoption(
        "--record-durations", action="store_true", default=False,
        help="merge this run's per-test durations into tests/durations.json.")
    parser.addini("tier_max_seconds", "duration threshold for --tier quick",
                  default="60")
    parser.addini("tier_budget_seconds",
                  "warn when a --tier quick run takes longer than this",
                  default="600")


def pytest_configure(config):
    known = {line.split(":", 1)[0].strip() for line in config.getini("markers")}
    if "slow" not in known:
        config.addinivalue_line(
            "markers", "slow: dropped by --tier quick (minutes-long or needs an oracle)")
    if "essential" not in known:
        config.addinivalue_line(
            "markers", "essential: kept by --tier quick whatever it costs")
    config._tiers = _TierState(config)
    config.pluginmanager.register(config._tiers, "pytest_tiers_state")


def _key(nodeid: str) -> str:
    """``tests/test_x.py::name`` whatever directory pytest was started from."""
    i = nodeid.find("tests/")
    return nodeid[i:] if i >= 0 else nodeid


def _load_durations() -> dict:
    try:
        return json.loads(DURATIONS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _git_head(path: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                                "--", "src", "tests"],
                               capture_output=True, text=True, timeout=30)
        sha = out.stdout.strip() or "unknown"
        return sha + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class _TierState:
    def __init__(self, config):
        self.config = config
        self.is_worker = hasattr(config, "workerinput")
        self.tier = config.getoption("tier")
        opt = config.getoption("tier_max")
        self.max_s = float(opt if opt is not None else config.getini("tier_max_seconds"))
        self.budget_s = float(config.getini("tier_budget_seconds"))
        self.data = _load_durations()
        self.known = {k: v["s"] for k, v in self.data.get("durations", {}).items()}
        self.measured: dict[str, float] = {}
        self.dropped_slow = self.dropped_long = self.kept_essential = 0
        self.selected: list[str] = []
        self.collected = False  # False on an xdist controller: its workers collect
        self.t0 = time.monotonic()
        self.flushed = self.t0
        self.run: dict | None = None   # this run's provenance, filled on first write

    # -- selection ------------------------------------------------------------
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        if self.tier == "quick":
            keep, drop = [], []
            for item in items:
                if item.get_closest_marker("essential"):
                    keep.append(item)
                    if (item.get_closest_marker("slow")
                            or self.known.get(_key(item.nodeid), 0.0) > self.max_s):
                        self.kept_essential += 1
                elif item.get_closest_marker("slow"):
                    drop.append(item)
                    self.dropped_slow += 1
                elif self.known.get(_key(item.nodeid), 0.0) > self.max_s:
                    drop.append(item)
                    self.dropped_long += 1
                else:
                    keep.append(item)
            if drop:
                config.hook.pytest_deselected(items=drop)
                items[:] = keep
        self.selected = [_key(item.nodeid) for item in items]
        self.collected = True

    # -- measurement ------------------------------------------------------------
    def pytest_runtest_logreport(self, report):
        if self.is_worker:      # xdist forwards every report to the controller
            return
        key = _key(report.nodeid)
        self.measured[key] = self.measured.get(key, 0.0) + report.duration
        # Flush as it goes: a recording interrupted after hours (a loaded box, a killed
        # job) keeps what it measured, and the next `make durations` fills the rest.
        if (report.when == "teardown" and self.config.getoption("record_durations")
                and time.monotonic() - self.flushed > _FLUSH_EVERY_S):
            self._write(final=False)

    def pytest_sessionfinish(self, session):
        if self.is_worker or not self.config.getoption("record_durations"):
            return
        self._write(final=True)

    def _write(self, final: bool):
        self.flushed = time.monotonic()
        if not self.measured:
            return
        today = date.today().isoformat()
        durations = dict(self.data.get("durations", {}))
        for nodeid, seconds in self.measured.items():
            durations[nodeid] = {"s": round(seconds, 3), "on": today}
        try:
            load = [round(x, 2) for x in os.getloadavg()]
        except OSError:
            load = None
        if self.run is None:
            self.run = {
                "date": today,
                "commit": _git_head(DURATIONS_FILE.parent),
                "host": socket.gethostname().split(".")[0],
                "machine": platform.machine(),
                "cpus": os.cpu_count(),
                "workers": getattr(self.config.option, "numprocesses", None) or 0,
                "args": [a for a in self.config.invocation_params.args
                         if not a.startswith("--record")],
            }
        self.run.update(load=load, tests=len(self.measured), complete=final)
        runs = self.data.get("runs", []) + [self.run]
        payload = {
            "about": "per-test setup+call+teardown seconds; an ordering for "
                     "`--tier quick`, not a benchmark (see pytest_tiers.py)",
            "runs": runs[-_KEEP_RUNS:],
            "durations": dict(sorted(durations.items())),
        }
        tmp = DURATIONS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1) + "\n")
        tmp.replace(DURATIONS_FILE)

    # -- reporting ------------------------------------------------------------
    def pytest_terminal_summary(self, terminalreporter):
        if self.is_worker:
            return
        wall = time.monotonic() - self.t0
        # Under xdist each worker selects for itself and the controller never sees the
        # collection, so it reports what ran instead of the drop breakdown.
        selected = self.selected if self.collected else list(self.measured)
        est = sum(self.known.get(n, 0.0) for n in selected)
        unknown = sum(1 for n in selected if n not in self.known)
        runs = self.data.get("runs", [])
        when = runs[-1]["date"] if runs else "never"
        line = f"tier {self.tier}: {len(selected)} selected"
        if self.tier == "quick" and self.collected:
            line += (f", dropped {self.dropped_slow} slow + {self.dropped_long} "
                     f"over {self.max_s:g} s, kept {self.kept_essential} essential")
        elif self.tier == "quick":
            line += f" (xdist: run without -n for the drop breakdown; max {self.max_s:g} s)"
        line += (f"; recorded estimate {est:.0f} s, {unknown} unmeasured; "
                 f"wall {wall:.0f} s; durations recorded {when}")
        terminalreporter.write_sep("-", line)
        if self.config.getoption("record_durations"):
            terminalreporter.write_line(
                f"recorded {len(self.measured)} durations -> {DURATIONS_FILE}")
        if self.tier == "quick" and wall > self.budget_s:
            terminalreporter.write_line(
                f"WARNING: the quick tier took {wall:.0f} s, over its "
                f"{self.budget_s:.0f} s budget. Re-record durations "
                f"(`make durations`) or lower --tier-max; see pytest_tiers.py.",
                yellow=True, bold=True)
