# Architecture References and License Compliance

This repository borrows **workflow ideas and interface patterns** from public projects and re-implements all logic in original code.
No large code sections were copied.

## Referenced Projects

1. `jtuhtan/hydro-sig` (GPL-3.0)
   - URL: https://github.com/jtuhtan/hydro-sig
   - Borrowed ideas:
     - Build a per-case hydrodynamic signature from multi-sensor time-series.
     - Concatenate per-sensor statistical/frequency descriptors into a single feature row.
     - User-guide style README for data and workflow.
   - Compliance:
     - Only high-level feature engineering concepts and data organization were reused.
     - No source file/code blocks were copied.

2. `Eliasfarah0/OpenFOAM-Automated-Tool-Chain-PoC` (MIT)
   - URL: https://github.com/Eliasfarah0/OpenFOAM-Automated-Tool-Chain-PoC
   - Borrowed ideas:
     - Python-driven OpenFOAM orchestration.
     - Stage-like toolchain (case generation -> mesh -> solve -> post-process).
     - Batch automation with logging and command wrappers.
   - Compliance:
     - Re-implemented in this repository’s own module layout.

3. `rmcconke/dataFoam` (MIT)
   - URL: https://github.com/rmcconke/dataFoam
   - Borrowed ideas:
     - Convert OpenFOAM outputs into Python-readable ML artifacts.
     - Explicit metadata and case indexing in dataset generation.
     - Consistent file naming and schema for downstream ML.
   - Compliance:
     - Re-implemented custom parsers for probe outputs and metadata indexing.

4. `nithinadidela/circular-cylinder` (license not declared in repository)
   - URL: https://github.com/nithinadidela/circular-cylinder
   - Borrowed ideas:
     - Minimal unsteady cylinder-case setup style and domain-level configuration references.
     - OpenFOAM case templating mindset for multiple Reynolds runs.
   - Compliance:
     - Only setup concept and case structure inspiration were used.
     - No direct code reuse from unlicensed files.

## Optional Contextual References

- `dynamicslab/wakeFoam` and `DiffPhys-CylinderWakeFlow` were considered for experiment organization style only.
- Their code was not imported into this repository.

## Summary

All implementation files in this repository are original and intended to be backend-swappable (`synthetic` <-> `openfoam`) while preserving one stable data schema:
- `data/raw/<case_id>/probes.csv`
- `data/raw/<case_id>/metadata.json`
- `data/raw/manifest.csv`
- `data/raw/index.csv`
