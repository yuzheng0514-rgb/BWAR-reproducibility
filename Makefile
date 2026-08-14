PYTHON ?= python

.PHONY: install test smoke s1 s2 s2-check simulations divvy pems figures

install:
	$(PYTHON) -m pip install -e .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

smoke:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py \
		--reps 1 --output-root results/generated/smoke/s1_geometry \
		--artifact-root artifacts/generated/smoke
	PYTHONPATH=src $(PYTHON) scripts/run_continuous_covariance_regimes.py \
		--config configs/continuous_covariance_regimes.json \
		--reps 1 --workers 1 --output results/generated/smoke/s2_continuous_covariance

s1:
	PYTHONPATH=src $(PYTHON) scripts/run_simulations.py --reps 80 --ar-model diag

s2:
	PYTHONPATH=src $(PYTHON) scripts/run_continuous_covariance_regimes.py \
		--config configs/continuous_covariance_regimes.json \
		--workers 1 --output results/generated/s2_continuous_covariance

s2-check:
	PYTHONPATH=src $(PYTHON) scripts/audit_continuous_covariance_regimes.py \
		--result-dir results/reference/s2_continuous_covariance

simulations: s1 s2

divvy:
	PYTHONPATH=src $(PYTHON) scripts/run_divvy.py

pems:
	PYTHONPATH=src $(PYTHON) scripts/run_pems_bay.py \
		--out results/generated/pems_bay

figures:
	PYTHONPATH=src $(PYTHON) scripts/build_artifacts.py
