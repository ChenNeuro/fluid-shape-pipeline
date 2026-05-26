PYTHON ?= $(shell command -v python || command -v python3)
CONFIG ?= configs/default.yaml
WORKERS ?=
SOLVER ?=
AUDIT_N_PERM ?=
RUN_DIR ?=
BACKBONE ?= resnet18
RUN_NAME ?=

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

ifeq ($(strip $(RUN_DIR)),)
  RUN_DIR_ARG :=
else
  RUN_DIR_ARG := --run-dir $(RUN_DIR)
endif

ifeq ($(strip $(RUN_NAME)),)
  TRAIN_RUN_NAME_ARG :=
  RECON_RUN_NAME_ARG := --run-name $(BACKBONE)
else
  TRAIN_RUN_NAME_ARG := --run-name $(RUN_NAME)
  RECON_RUN_NAME_ARG := --run-name $(RUN_NAME)
endif

.PHONY: dataset train sota reconstruct audit figure gif report wake-dataset wake-fields wake-audit wake-train wake-reconstruct wake-pipeline clean

dataset:
	$(PYTHON) -m sim.generate_dataset --config $(CONFIG) $(SOLVER_ARG) $(WORKERS_ARG) $(RUN_DIR_ARG)
	$(PYTHON) -m extract.build_features --config $(CONFIG) $(RUN_DIR_ARG)

train:
	$(PYTHON) -m ml.train --config $(CONFIG) $(RUN_DIR_ARG)

sota:
	$(PYTHON) -m ml.train_sota --config $(CONFIG) $(RUN_DIR_ARG)

reconstruct:
	$(PYTHON) -m ml.reconstruct --config $(CONFIG) $(RUN_DIR_ARG)

audit:
	$(PYTHON) -m ml.audit_shortcut --config $(CONFIG) $(AUDIT_PERM_ARG) $(RUN_DIR_ARG)

figure:
	$(PYTHON) scripts/make_publication_figure.py --config $(CONFIG) --output reports/figure_main_reproducible.png

gif:
	$(PYTHON) scripts/make_report_gif.py --reports-dir reports --output reports/pipeline_overview.gif

report: train

wake-dataset:
	$(PYTHON) -m sim.generate_dataset --config $(CONFIG) $(SOLVER_ARG) $(WORKERS_ARG) $(RUN_DIR_ARG)

wake-fields:
	$(PYTHON) -m extract.build_wake_fields --config $(CONFIG) $(RUN_DIR_ARG)

wake-audit:
	$(PYTHON) -m ml.audit_wake_leakage --config $(CONFIG) $(RUN_DIR_ARG)

wake-train:
	$(PYTHON) -m ml.train_wake --config $(CONFIG) --backbone $(BACKBONE) $(TRAIN_RUN_NAME_ARG) $(RUN_DIR_ARG)

wake-reconstruct:
	$(PYTHON) -m ml.reconstruct_wake --config $(CONFIG) $(RECON_RUN_NAME_ARG) $(RUN_DIR_ARG)

wake-pipeline: wake-dataset wake-fields wake-train wake-reconstruct

clean:
	rm -rf data/raw/* data/features/features.csv data/wake_fields/* reports/* models/* logs/*.log runs/openfoam/*
