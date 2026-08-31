# edge-ai-architect
#
# Disk binds on this machine, so environments are built per phase and torn down
# when the phase ends rather than accumulating every backend in one venv.
# `make disk` before starting a phase; `make clean-phase` when finishing one.
# The env-* targets use `uv venv --clear`, so re-running one is safe: your work
# lives in projects/, notes/ and results/, never in the venv.

VENV        ?= .venv
PY          := $(VENV)/bin/python
UV          ?= uv
MODELS_DIR  ?= models
BUDGET_GB   ?= 8

.DEFAULT_GOAL := help

# ---------------------------------------------------------------- environments

.PHONY: env-core
env-core: ## Core only: edgebench + psutil, no ML frameworks (~15 MB)
	$(UV) venv --clear --python 3.10 $(VENV)
	$(UV) pip install -e ".[dev]"

.PHONY: env-p01
env-p01: ## Phase 1: training + compression (torch, torchvision, viz)
	$(UV) venv --clear --python 3.10 $(VENV)
	$(UV) pip install -e ".[torch,viz,dev]"

.PHONY: env-p02
env-p02: ## Phase 2: portable runtimes (torch, onnx, optimum, viz)
	$(UV) venv --clear --python 3.10 $(VENV)
	$(UV) pip install -e ".[torch,onnx,viz,dev]"

.PHONY: env-p03
env-p03: ## Phase 3: NLP foundations (torch, HuggingFace stack, sklearn, viz)
	$(UV) venv --clear --python 3.10 $(VENV)
	$(UV) pip install -e ".[torch,hf,nlp,viz,dev]"

.PHONY: env-p04
env-p04: ## Phase 4: on-device LLMs (torch, HuggingFace stack, viz)
	$(UV) venv --clear --python 3.10 $(VENV)
	$(UV) pip install -e ".[torch,hf,viz,dev]"

.PHONY: env-apple
env-apple: ## Add the optional Darwin-only comparison backends (coremltools, mlx)
	$(UV) pip install -e ".[apple]"

# ------------------------------------------------------------------- verifying

.PHONY: check-venv
check-venv:
	@test -x $(PY) || { \
		echo "No phase environment found at $(VENV)/"; \
		echo; \
		echo "Create one first:"; \
		echo "  make env-core   # just edgebench, no ML frameworks (~15 MB)"; \
		echo "  make env-p01    # Phase 1: training + compression"; \
		echo "  make help       # all options"; \
		exit 1; }

.PHONY: selftest
selftest: check-venv ## Run the Phase 0 acceptance test
	$(PY) -m edgebench.selftest

.PHONY: check
check: check-venv ## Prove ground rule 2 still holds: same harness, different devices
	@echo "== auto-selected device =="
	@$(PY) -m edgebench.selftest --no-store
	@echo
	@echo "== forced cpu =="
	@EDGEBENCH_DEVICE=cpu $(PY) -m edgebench.selftest --no-store
	@echo
	@echo "== power sampling disabled =="
	@EDGEBENCH_POWER=null $(PY) -m edgebench.selftest --no-store
	@echo
	@echo "== backend availability =="
	@$(PY) -c "import runners; print(runners.describe())"

.PHONY: report
report: check-venv ## Print the results table
	$(PY) -m edgebench.report

.PHONY: pareto
pareto: check-venv ## Plot quality against latency (needs the viz extra)
	$(PY) -m edgebench.report --plot results/pareto.png

.PHONY: test
test: check-venv ## Run the unit tests
	$(PY) -m pytest -q

# --------------------------------------------------------------------- hygiene

.PHONY: disk
disk: ## Show free disk and the model cache against its budget
	@echo "Filesystem:"
	@df -h /System/Volumes/Data 2>/dev/null | tail -1 || df -h / | tail -1
	@echo
	@printf "Model cache (%s): " "$(MODELS_DIR)"
	@du -sh $(MODELS_DIR) 2>/dev/null | cut -f1 || echo "0B"
	@echo "Budget: $(BUDGET_GB) GB"
	@echo
	@echo "Largest cached files:"
	@found=$$(find $(MODELS_DIR) -type f -size +50M -exec du -h {} + 2>/dev/null | sort -rh | head -10); \
	if [ -n "$$found" ]; then echo "$$found"; else echo "  (none over 50 MB)"; fi

.PHONY: clean-phase
clean-phase: ## Delete the venv and report reclaimed space
	@before=$$(du -sm $(VENV) 2>/dev/null | cut -f1 || echo 0); \
	rm -rf $(VENV); \
	echo "removed $(VENV), reclaimed $${before} MB"

.PHONY: clean
clean: ## Remove caches and build artifacts (keeps venv, models, results)
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	@echo "caches cleared"

.PHONY: help
help: ## List targets
	@echo "edge-ai-architect"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Start here:  make env-core && make selftest"
