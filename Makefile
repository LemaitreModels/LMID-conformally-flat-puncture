PY ?= python

# `caffeinate` keeps macOS awake across a run measured in tens of minutes.  It
# does not exist on Linux, where an unguarded `caffeinate -i` makes `make test`
# die with "command not found" before pytest is ever reached — so the target
# fails on exactly the machines (CI, the cluster) that cannot sleep anyway.
# Resolve it once, to the empty string when absent.
NOSLEEP := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i)

.PHONY: install test test-quick fast figdata figures tabdata tables models oracle clean

install:
	pip install -e ".[dev]" --config-settings editable_mode=compat

# Three tiers, and the middle one is the point.  Measured 2026-08-16 on leaf
# 0fcaafe, 8-core M-series laptop at `uptime` load 2.4-5.4:
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
	$(NOSLEEP) $(PY) -m pytest -q

# The development loop for anything that touches the solver or the ROM: every
# test except the 30 that need an external oracle or are minutes-long by nature.
# 4.5x faster than the full suite for 95 % of the coverage.
#
# It is not a substitute for `make test` before a commit — see the marker note in
# pyproject.toml for what `slow` is and is not allowed to mean.
test-quick:
	$(NOSLEEP) $(PY) -m pytest -q -m "not slow"

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
	@echo "Build the external TwoPunctures oracle binary. See docs/DATA.md."

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
