# Abaqus FEM validation

This directory contains the finite-element reference models used to validate
the analytical HOST implementation and other sandwich theories.

## Structure

- `builders/` - Abaqus model builders
- `materials/` - TPMS homogenized-material database access and transformations
- `postprocessing/` - ODB extraction and standardized comparison outputs
- `common/` - shared utilities
- `cases/` - reproducible FEM case definitions

## Generated simulation data

Raw Abaqus simulation data are not stored in this repository.

Default computational workspace:

    C:\AbaqusScratch\HOST

Current simulation families:

    C:\AbaqusScratch\HOST\4pb_homogenized
    C:\AbaqusScratch\HOST\4pb_explicit
    C:\AbaqusScratch\HOST\buckling_homogenized
    C:\AbaqusScratch\HOST\buckling_explicit

Machine-specific paths are configured through:

    config\paths.local.json

This file is intentionally excluded from Git.

## Run-directory hygiene

Homogenized four-point-bending runs use deterministic case directories of the
form:

    <topology>_rd<...>_<load>_<state>_nh<...>_..._<element>_<hash>

The hash is generated from the result-relevant model configuration, including
the selected database state, rotated constitutive tensor, geometry, mesh,
loading, material properties and orientation.

Existing non-empty case directories are protected from accidental overwrite by
default.

Each run writes a `manifest.json` containing the complete case definition,
database provenance, coordinate transformation, Git commit/status and SHA-256
hashes of the builder and material-query scripts.
