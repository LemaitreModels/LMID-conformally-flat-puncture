PY ?= python

.PHONY: install test figdata figures models oracle clean

install:
	pip install -e ".[dev]"

test:
	caffeinate -i pytest -q

# --- figures: regenerate data (recompute from the solver/ROM), then plot ---
figdata:
	$(PY) manuscript/figures/make_figdata.py --all

figures: figdata
	cd manuscript/figures && for f in fig??_*_plot.py; do echo ">> $$f"; $(PY) "$$f"; done

# --- heavy tier (documented, mostly cluster) ---
models:
	@echo "Heavy chi surrogate build (cluster). See docs/DATA.md."
	@echo "  pipeline/models/{build_surrogate_chi,run_8d_chi_array,build_pod_hermite_model_chi,...}.py"

oracle:
	@echo "Build the external TwoPunctures oracle binary. See docs/DATA.md."

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
