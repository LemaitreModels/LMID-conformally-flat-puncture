PY ?= python

# `caffeinate` keeps macOS awake across a run measured in tens of minutes.  It
# does not exist on Linux, where an unguarded `caffeinate -i` makes `make test`
# die with "command not found" before pytest is ever reached — so the target
# fails on exactly the machines (CI, the cluster) that cannot sleep anyway.
# Resolve it once, to the empty string when absent.
NOSLEEP := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i)

# Parallel workers for `test`, `test-quick` and `durations` (pytest-xdist). Four, not
# one per core: concurrent JAX solves run out of memory before they run out of cores,
# and peer sessions share this box. `--dist loadfile` keeps a file on one worker, so a
# module-scoped fixture is built once rather than once per worker. Override with
# `make test NPROC=1` (or use `test-serial`) when the gates' printed numbers matter:
# xdist swallows -s output.
NPROC ?= 4
XDIST := -n $(NPROC) --dist loadfile

.PHONY: install test test-serial test-quick dev durations fast figdata figures tabdata tables models oracle docs clean

install:
	pip install -e ".[dev]" --config-settings editable_mode=compat

# Three tiers, and the middle one is the point.  Measured 2026-08-16 on leaf
# 3f4462d, 8-core M-series laptop at `uptime` load 2.4-5.4:
#
#     make fast        241 tests      ~1 s      structural invariants only
#     make test-quick  631 tests     329 s      95 % of the tests, 22 % of the clock
#     make test        661 tests    1497 s      everything
#
# Treat those as measurements, not constants: the suite is dominated by elliptic
# solves and the figure has moved with the solver, the test count and the load on
# the box.  Re-measure before quoting one.

# The FULL acceptance suite.  Background it; do not pipe it through head/tail,
# which buffers until pytest exits and makes a live run look hung.
#
# Five tests are 75 % of this: test_joint_held_out_below_1e8 alone is 658 s (44 %),
# then test_certified_polish_4d 140 s, test_validation_3d's TwoPunctures fixture
# 128 s, test_tp_cross_check_3d 126 s and test_spin_axisymmetry_nphi 71 s.  Any
# future attempt to cut this number has to start there.
test:
	$(NOSLEEP) $(PY) -m pytest -q $(XDIST)

# The same suite in one process, for reading what the gates print (-s).
test-serial:
	$(NOSLEEP) $(PY) -m pytest -q -s

# The development loop for anything that touches the solver or the ROM: every
# test except those marked `slow` and those recorded slower than 60 s in
# tests/durations.json, but never a test marked `essential` (tests/pytest_tiers.py).
# Until 2026-10-17 this was `-m "not slow"` alone, 4.5x faster than the full suite for
# 95 % of the coverage; the duration threshold keeps it from drifting as tests are
# added, and the run warns when it exceeds its 600 s budget.
#
# It is not a substitute for `make test` before a commit — see the marker note in
# pyproject.toml for what `slow` is and is not allowed to mean.
test-quick:
	$(NOSLEEP) $(PY) -m pytest -q --tier quick $(XDIST)

# The edit loop: only the quick-tier tests whose executed lines changed since the
# last `make dev` (pytest-testmon; its database is .testmondata, gitignored). The
# first run executes the whole quick tier. Blind to data files, committed artifacts
# and edits in sibling leaves, so it never replaces `make test` before a commit.
dev:
	$(NOSLEEP) $(PY) -m pytest -q --testmon --tier quick

# Re-measure every test and rewrite tests/durations.json, which `--tier quick` reads.
# Commit the result. Re-run when a test is added or a solve changes cost markedly.
durations:
	$(NOSLEEP) $(PY) -m pytest -q $(XDIST) --record-durations

# The structural tier: seconds, no solves, and the CI job.  It covers the
# invariants a refactor breaks silently — standalone-ness and namespace ownership
# (test_self_containment), the single-sourced certification gate
# (test_certification) and the QC wiring (test_qc_wiring).
fast:
	$(NOSLEEP) $(PY) -m pytest -q tests/test_self_containment.py \
	                              tests/test_certification.py tests/test_qc_wiring.py

# --- figures: distil any MISSING figdata from the raw run artifacts, then plot ---
# NOTE: --all skips a figdata that is already present, and all ten are committed,
# so on a clone this rebuilds nothing and `figures` simply replots.  Use
# `make_figdata.py --all --force` to actually re-distil (needs the heavy tier).
figdata:
	$(PY) paper/figures/make_figdata.py --all

# `|| exit 1` is load-bearing: a bare `for` loop reports only its LAST command's
# status, so without it every plotter but the tenth could die unnoticed.
figures: figdata
	cd paper/figures && for f in fig??_*_plot.py; do echo ">> $$f"; $(PY) "$$f" || exit 1; done

# --- tables: recompute data from the solver, then render the LaTeX bodies ---
tabdata:
	$(PY) paper/tables/make_tabdata.py --all

tables: tabdata
	cd paper/tables && for f in tab??_*_tex.py; do echo ">> $$f"; $(PY) "$$f" || exit 1; done

# --- heavy tier (documented, mostly cluster) ---
models:
	@echo "Heavy chi surrogate build (cluster). See docs/DATA.md."
	@echo "  python -m lemaitre.initial_data.conformally_flat_puncture.pipeline.<builder>"
	@echo "  builders: build_surrogate_chi, run_8d_chi_array, build_pod_hermite_model_chi,"
	@echo "            build_pod_hermite_model_chi_8d, build_pod_hermite_chi8d_array,"
	@echo "            build_cross_model_chi"

oracle:
	@bash oracle/build.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# The documentation: HTML into docs/_site/ (gitignored). Needs the [docs] extra.
docs:
	$(PY) -m sphinx -b html -j auto docs docs/_site
