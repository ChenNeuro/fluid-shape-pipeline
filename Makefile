PYTHON ?= $(shell command -v python || command -v python3)
CONFIG ?= configs/default.yaml
WORKERS ?=
SOLVER ?=
AUDIT_N_PERM ?=

ifeq ($(strip $(WORKERS)),)
  WORKERS_ARG :=
else
  WORKERS_ARG := --workers $(WORKERS)
endif

ifeq ($(strip $(SOLVER)),)
  SOLVER_ARG :=
else
  SOLVER_ARG := --solver $(SOLVER)
endif

ifeq ($(strip $(AUDIT_N_PERM)),)
  AUDIT_PERM_ARG :=
else
  AUDIT_PERM_ARG := --n_perm $(AUDIT_N_PERM)
endif

.PHONY: dataset train sota reconstruct audit figure gif report clean

dataset:
	$(PYTHON) -m sim.generate_dataset --config $(CONFIG) $(SOLVER_ARG) $(WORKERS_ARG)
	$(PYTHON) -m extract.build_features --config $(CONFIG)

train:
	$(PYTHON) -m ml.train --config $(CONFIG)

sota:
	$(PYTHON) -m ml.train_sota --config $(CONFIG)

reconstruct:
	$(PYTHON) -m ml.reconstruct --config $(CONFIG)

audit:
	$(PYTHON) -m ml.audit_shortcut --config $(CONFIG) $(AUDIT_PERM_ARG)

figure:
	$(PYTHON) scripts/make_publication_figure.py --config $(CONFIG) --output reports/figure_main_reproducible.png

gif:
	$(PYTHON) scripts/make_report_gif.py --reports-dir reports --output reports/pipeline_overview.gif

report: train

clean:
	rm -rf data/raw/* data/features/features.csv reports/* models/* logs/*.log runs/openfoam/*
