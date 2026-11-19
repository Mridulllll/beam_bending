# -*- coding: utf-8 -*-
r"""
Independent Abaqus constitutive-semantics audit for the HOST homogenized core.

Purpose
-------
Verify, with six affine single-element tests, that the exact 21 constants used
by the 4PB builder make Abaqus/Standard reproduce

    sigma = C_hom_model * epsilon_eng

for the canonical engineering-shear Voigt order

    [xx, yy, zz, xy, xz, yz].

This independently checks the three assumptions that matter before using the
3D model as a dissertation reference:

1. the Abaqus fully-anisotropic 21-value table ordering;
2. engineering-shear semantics;
3. the GLOBAL material-orientation assignment (no second RVE rotation).

The script reads the already committed source_case_manifest.json so it audits
the exact tensor and exact 21 constants used by the selected 4PB case.

Run from the HOST repository, for example:

    $env:HOST_4PB_CONSTITUTIVE_MANIFEST = `
      ".\reference_results\4pb\Gyroid_rd12_initial_C3D20R_ne4_fz4_42665a80e1\source_case_manifest.json"

    abaqus cae noGUI="fem\abaqus\verification\audit_anisotropic_core_semantics.py"

    Remove-Item Env:HOST_4PB_CONSTITUTIVE_MANIFEST

Optional:
    HOST_4PB_CONSTITUTIVE_AUDIT_DIR
    HOST_4PB_CONSTITUTIVE_STRAIN=1e-4
    HOST_4PB_CONSTITUTIVE_CPUS=1

Abaqus/Python 3.8 compatible; no NumPy/Pandas required.
"""

from __future__ import print_function

import json
import math
import os
import shutil
import sys

from abaqus import mdb
# Abaqus/CAE extends the core object model lazily through its CAE modules.
# Without this import, objects such as Model can exist while step/output/BC
# repositories and constructors (e.g. StaticStep, FieldOutputRequest) are not
# attached to the Model object in noGUI execution.
from caeModules import *

# ``from caeModules import *`` can shadow ordinary Python built-ins such as
# ``sum`` with Abaqus command objects. Keep an explicit handle to the genuine
# Python built-ins for numerical reductions used by this audit.
import builtins as _py_builtins

from abaqusConstants import (
    ANALYSIS,
    ANISOTROPIC,
    AXIS_3,
    CARTESIAN,
    C3D8,
    DEFORMABLE_BODY,
    FULL,
    GLOBAL,
    OFF,
    ON,
    PERCENTAGE,
    ROTATION_NONE,
    STACK_3,
    STANDARD,
    THREE_D,
)
import mesh
from odbAccess import openOdb


SCHEMA = "host_abaqus_anisotropic_constitutive_audit_v1"

VOIGT = (
    "xx",
    "yy",
    "zz",
    "xy",
    "xz",
    "yz",
)

STRAIN_CASES = (
    "EXX",
    "EYY",
    "EZZ",
    "GXY",
    "GXZ",
    "GYZ",
)


def env_required(name):
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(
            "Required environment variable {} is not set.".format(name)
        )
    return os.path.abspath(os.path.expanduser(value.strip()))


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return float(default)
    return float(value)


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return int(default)
    return int(value)


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def matvec(C, vector):
    return [
        _py_builtins.sum(
            float(C[i][j]) * float(vector[j])
            for j in range(6)
        )
        for i in range(6)
    ]


def vector_norm_inf(vector):
    return _py_builtins.max(abs(float(value)) for value in vector)


def vector_sub(a, b):
    return [
        float(a[i]) - float(b[i])
        for i in range(len(a))
    ]


def vector_average(vectors):
    if not vectors:
        raise RuntimeError("Cannot average an empty vector list.")

    n = float(len(vectors))
    width = len(vectors[0])

    return [
        _py_builtins.sum(float(vector[j]) for vector in vectors) / n
        for j in range(width)
    ]


def symmetric_relative_matrix_error(C):
    max_c = _py_builtins.max(
        abs(float(C[i][j]))
        for i in range(6)
        for j in range(6)
    )
    max_diff = _py_builtins.max(
        abs(float(C[i][j]) - float(C[j][i]))
        for i in range(6)
        for j in range(6)
    )
    return max_diff / _py_builtins.max(max_c, 1.0)


def relative_matrix_inf_error(A, B):
    scale = _py_builtins.max(
        abs(float(B[i][j]))
        for i in range(6)
        for j in range(6)
    )
    diff = _py_builtins.max(
        abs(float(A[i][j]) - float(B[i][j]))
        for i in range(6)
        for j in range(6)
    )
    return diff / _py_builtins.max(scale, 1.0e-30)


def abaqus_21_to_matrix(values):
    """
    Convert the Abaqus fully-anisotropic 21-value upper-triangular ordering to
    the HOST canonical engineering-shear matrix order
    [xx, yy, zz, xy, xz, yz].
    """
    if len(values) != 21:
        raise RuntimeError(
            "Expected 21 anisotropic constants; got {}.".format(len(values))
        )

    v = [float(value) for value in values]

    C = [[0.0] * 6 for _ in range(6)]

    mapping = (
        (0, 0),                 # C11
        (0, 1), (1, 1),         # C12, C22
        (0, 2), (1, 2), (2, 2), # C13, C23, C33
        (0, 3), (1, 3), (2, 3), (3, 3),
        (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),
        (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5),
    )

    for value, (i, j) in zip(v, mapping):
        C[i][j] = value
        C[j][i] = value

    return C


def read_anisotropic_21_from_inp(inp_path):
    """
    Read the actual constants serialized by Abaqus/CAE into the input deck.

    CAE may round material constants when writing the .inp. Comparing the ODB
    response against these serialized constants separates constitutive
    semantics from harmless text-deck precision loss.
    """
    with open(inp_path, "r") as handle:
        lines = handle.readlines()

    collecting = False
    values = []

    for line in lines:
        stripped = line.strip()

        if not collecting:
            upper = stripped.upper()
            if (
                upper.startswith("*ELASTIC")
                and "ANISOTROPIC" in upper
            ):
                collecting = True
            continue

        if stripped.startswith("*"):
            break

        if not stripped:
            continue

        for token in stripped.split(","):
            token = token.strip()
            if token:
                values.append(float(token))

    if len(values) < 21:
        raise RuntimeError(
            "{}: expected at least 21 anisotropic constants after *Elastic; "
            "found {}.".format(inp_path, len(values))
        )

    return values[:21]


def target_engineering_strain(case_name, magnitude):
    strain = [0.0] * 6
    index = STRAIN_CASES.index(case_name)
    strain[index] = float(magnitude)
    return strain


def affine_displacement(x, y, z, engineering_strain):
    exx, eyy, ezz, gxy, gxz, gyz = engineering_strain

    # Symmetric displacement gradient; engineering shear gamma_ij = 2*eps_ij.
    u1 = (
        exx * x
        + 0.5 * gxy * y
        + 0.5 * gxz * z
    )

    u2 = (
        0.5 * gxy * x
        + eyy * y
        + 0.5 * gyz * z
    )

    u3 = (
        0.5 * gxz * x
        + 0.5 * gyz * y
        + ezz * z
    )

    return (
        float(u1),
        float(u2),
        float(u3),
    )


def clean_job_files(workdir, job_name):
    prefixes = (
        job_name + ".",
        job_name + "_",
    )

    for name in os.listdir(workdir):
        if any(name.startswith(prefix) for prefix in prefixes):
            path = os.path.join(workdir, name)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def field_value_components(field_value):
    """
    Return numeric FieldValue components independent of ODB precision.

    Abaqus exposes single-precision field output through ``data`` and
    double-precision field output through ``dataDouble``. Accessing the wrong
    property raises an OdbError, so try the double-precision interface first
    and fall back to the ordinary interface.
    """
    try:
        raw = field_value.dataDouble
        return tuple(float(value) for value in raw)
    except Exception:
        raw = field_value.data
        return tuple(float(value) for value in raw)


def build_and_run_case(
    case_name,
    C_model,
    abaqus_constants,
    strain_magnitude,
    workdir,
    cpus,
):
    model_name = "MODEL_{}".format(case_name)
    job_name = "audit_{}".format(case_name.lower())

    if model_name in mdb.models:
        del mdb.models[model_name]

    if job_name in mdb.jobs:
        del mdb.jobs[job_name]

    clean_job_files(workdir, job_name)

    model = mdb.Model(name=model_name)

    sketch = model.ConstrainedSketch(
        name="__profile__",
        sheetSize=5.0,
    )

    sketch.rectangle(
        point1=(0.0, 0.0),
        point2=(1.0, 1.0),
    )

    part = model.Part(
        name="CUBE",
        dimensionality=THREE_D,
        type=DEFORMABLE_BODY,
    )

    part.BaseSolidExtrude(
        sketch=sketch,
        depth=1.0,
    )

    del model.sketches["__profile__"]

    part.Set(
        name="SET_ALL",
        cells=part.cells,
    )

    material = model.Material(
        name="MAT_AUDIT"
    )

    material.Elastic(
        type=ANISOTROPIC,
        table=(tuple(float(value) for value in abaqus_constants),),
    )

    model.HomogeneousSolidSection(
        name="SEC_AUDIT",
        material="MAT_AUDIT",
        thickness=None,
    )

    part.SectionAssignment(
        region=part.sets["SET_ALL"],
        sectionName="SEC_AUDIT",
    )

    part.MaterialOrientation(
        region=part.sets["SET_ALL"],
        orientationType=GLOBAL,
        axis=AXIS_3,
        additionalRotationType=ROTATION_NONE,
        angle=0.0,
        localCsys=None,
        fieldName="",
        stackDirection=STACK_3,
    )

    part.seedPart(
        size=1.0,
        deviationFactor=0.1,
        minSizeFactor=0.1,
    )

    elem_type = mesh.ElemType(
        elemCode=C3D8,
        elemLibrary=STANDARD,
    )

    part.setElementType(
        regions=(part.cells,),
        elemTypes=(elem_type,),
    )

    part.generateMesh()

    if len(part.elements) != 1:
        raise RuntimeError(
            "{}: expected exactly one C3D8 element, got {}.".format(
                case_name,
                len(part.elements),
            )
        )

    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)

    instance = assembly.Instance(
        name="CUBE-1",
        part=part,
        dependent=ON,
    )

    assembly.regenerate()

    model.StaticStep(
        name="STRAIN",
        previous="Initial",
        nlgeom=OFF,
        timePeriod=1.0,
        initialInc=1.0,
        minInc=1.0e-8,
        maxInc=1.0,
        maxNumInc=10,
    )

    # Request only what the audit needs. Do not try to enumerate/delete
    # model.fieldOutputRequests here: some Abaqus/CAE releases expose the
    # FieldOutputRequest constructor but not that repository as a direct Model
    # attribute in noGUI scripting. A second request is harmless for this
    # one-element audit, and the ODB reader explicitly accesses S and E.
    model.FieldOutputRequest(
        name="F_AUDIT",
        createStepName="STRAIN",
        variables=("S", "E", "U", "RF"),
        frequency=1,
    )

    target_strain = target_engineering_strain(
        case_name,
        strain_magnitude,
    )

    for node in instance.nodes:
        x, y, z = (
            float(node.coordinates[0]),
            float(node.coordinates[1]),
            float(node.coordinates[2]),
        )

        u1, u2, u3 = affine_displacement(
            x,
            y,
            z,
            target_strain,
        )

        set_name = "NODE_{}".format(int(node.label))

        assembly.SetFromNodeLabels(
            name=set_name,
            nodeLabels=(
                (
                    instance.name,
                    (int(node.label),),
                ),
            ),
        )

        model.DisplacementBC(
            name="BC_{}".format(int(node.label)),
            createStepName="STRAIN",
            region=assembly.sets[set_name],
            u1=u1,
            u2=u2,
            u3=u3,
        )

    job = mdb.Job(
        name=job_name,
        model=model_name,
        description=(
            "HOST anisotropic constitutive semantics audit: {}".format(
                case_name
            )
        ),
        type=ANALYSIS,
        numCpus=int(cpus),
        numDomains=int(cpus),
        memory=90,
        memoryUnits=PERCENTAGE,
        nodalOutputPrecision=FULL,
    )

    job.submit()
    job.waitForCompletion()

    inp_path = os.path.join(
        workdir,
        job_name + ".inp",
    )

    deck_constants = read_anisotropic_21_from_inp(
        inp_path
    )
    C_deck = abaqus_21_to_matrix(
        deck_constants
    )

    odb_path = os.path.join(
        workdir,
        job_name + ".odb",
    )

    if not os.path.exists(odb_path):
        raise RuntimeError(
            "{} did not produce {}.".format(
                job_name,
                odb_path,
            )
        )

    odb = openOdb(
        path=odb_path,
        readOnly=True,
    )

    try:
        step = odb.steps["STRAIN"]
        frame = step.frames[-1]

        if "S" not in frame.fieldOutputs:
            raise RuntimeError("{}: S output missing.".format(case_name))

        if "E" not in frame.fieldOutputs:
            raise RuntimeError("{}: E output missing.".format(case_name))

        stress_values = [
            field_value_components(field_value)
            for field_value in frame.fieldOutputs["S"].values
        ]

        strain_values = [
            field_value_components(field_value)
            for field_value in frame.fieldOutputs["E"].values
        ]

        stress_avg = vector_average(stress_values)
        strain_avg = vector_average(strain_values)

        expected_from_measured_strain = matvec(
            C_model,
            strain_avg,
        )

        expected_from_target_strain = matvec(
            C_model,
            target_strain,
        )

        expected_from_deck_measured_strain = matvec(
            C_deck,
            strain_avg,
        )

        source_stress_error = vector_sub(
            stress_avg,
            expected_from_measured_strain,
        )

        deck_stress_error = vector_sub(
            stress_avg,
            expected_from_deck_measured_strain,
        )

        strain_error = vector_sub(
            strain_avg,
            target_strain,
        )

        source_stress_scale = _py_builtins.max(
            vector_norm_inf(expected_from_measured_strain),
            1.0e-12,
        )

        deck_stress_scale = _py_builtins.max(
            vector_norm_inf(expected_from_deck_measured_strain),
            1.0e-12,
        )

        strain_scale = _py_builtins.max(
            abs(float(strain_magnitude)),
            1.0e-30,
        )

        return {
            "case": case_name,
            "target_engineering_strain": target_strain,
            "abaqus_average_engineering_strain": strain_avg,
            "strain_error": strain_error,
            "strain_inf_relative_to_applied_magnitude": (
                vector_norm_inf(strain_error) / strain_scale
            ),
            "abaqus_average_stress_MPa": stress_avg,
            "expected_stress_from_measured_E_MPa": (
                expected_from_measured_strain
            ),
            "expected_stress_from_target_strain_MPa": (
                expected_from_target_strain
            ),
            "serialized_deck_constants_MPa": deck_constants,
            "serialized_deck_C_MPa": C_deck,
            "expected_stress_from_serialized_deck_MPa": (
                expected_from_deck_measured_strain
            ),
            "source_stress_error_MPa": source_stress_error,
            "source_stress_inf_relative_error": (
                vector_norm_inf(source_stress_error)
                / source_stress_scale
            ),
            "deck_stress_error_MPa": deck_stress_error,
            "deck_stress_inf_relative_error": (
                vector_norm_inf(deck_stress_error)
                / deck_stress_scale
            ),
            "deck_vs_source_C_inf_relative_error": (
                relative_matrix_inf_error(
                    C_deck,
                    C_model,
                )
            ),
            # Backward-compatible aliases from v1-v5. These intentionally
            # refer to the source high-precision tensor comparison.
            "stress_error_MPa": source_stress_error,
            "stress_inf_relative_error": (
                vector_norm_inf(source_stress_error)
                / source_stress_scale
            ),
            "n_stress_integration_points": len(stress_values),
            "n_strain_integration_points": len(strain_values),
            "odb": odb_path,
        }
    finally:
        odb.close()


def main():
    manifest_path = env_required(
        "HOST_4PB_CONSTITUTIVE_MANIFEST"
    )

    if not os.path.isfile(manifest_path):
        raise IOError(
            "Manifest does not exist: {}".format(manifest_path)
        )

    manifest = load_json(manifest_path)

    constitutive = manifest["constitutive"]

    C_model = [
        [float(value) for value in row]
        for row in constitutive["C_hom_model_MPa"]
    ]

    abaqus_constants = [
        float(value)
        for value in constitutive[
            "abaqus_anisotropic_21_constants_MPa"
        ]
    ]

    if len(C_model) != 6 or any(len(row) != 6 for row in C_model):
        raise RuntimeError(
            "constitutive.C_hom_model_MPa must be 6x6."
        )

    if len(abaqus_constants) != 21:
        raise RuntimeError(
            "Expected 21 Abaqus anisotropic constants; got {}."
            .format(len(abaqus_constants))
        )

    voigt = tuple(
        str(item).lower()
        for item in constitutive["canonical_voigt_order"]
    )

    if voigt != VOIGT:
        raise RuntimeError(
            "Audit requires canonical Voigt {}; manifest has {}."
            .format(
                list(VOIGT),
                list(voigt),
            )
        )

    shear_convention = str(
        constitutive.get(
            "shear_convention",
            "",
        )
    ).lower()

    if shear_convention != "engineering":
        raise RuntimeError(
            "Audit requires engineering shear convention; manifest has {!r}."
            .format(shear_convention)
        )

    strain_magnitude = env_float(
        "HOST_4PB_CONSTITUTIVE_STRAIN",
        1.0e-4,
    )

    cpus = env_int(
        "HOST_4PB_CONSTITUTIVE_CPUS",
        1,
    )

    case_hash = str(
        manifest["case"]["case_hash"]
    )

    workdir = os.environ.get(
        "HOST_4PB_CONSTITUTIVE_AUDIT_DIR"
    )

    if workdir is None or not workdir.strip():
        workdir = os.path.join(
            r"C:\AbaqusScratch\HOST",
            "constitutive_audit_{}".format(
                case_hash
            ),
        )

    workdir = os.path.abspath(
        os.path.expanduser(
            workdir.strip()
        )
    )

    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    old_cwd = os.getcwd()
    os.chdir(workdir)

    try:
        print("=" * 88)
        print("HOST / ABAQUS ANISOTROPIC CONSTITUTIVE SEMANTICS AUDIT")
        print("=" * 88)
        print("Manifest       : {}".format(manifest_path))
        print("Case hash      : {}".format(case_hash))
        print("Work directory : {}".format(workdir))
        print("Voigt order    : {}".format(list(VOIGT)))
        print("Shear          : engineering")
        print("Orientation    : GLOBAL, zero additional rotation")
        print("Applied strain : {:.8g}".format(strain_magnitude))
        print(
            "C symmetry err : {:.3e}".format(
                symmetric_relative_matrix_error(C_model)
            )
        )
        print()

        results = []

        for case_name in STRAIN_CASES:
            print("Running {} ...".format(case_name))

            result = build_and_run_case(
                case_name=case_name,
                C_model=C_model,
                abaqus_constants=abaqus_constants,
                strain_magnitude=strain_magnitude,
                workdir=workdir,
                cpus=cpus,
            )

            results.append(result)

            print(
                "  strain rel.err = {:.3e}".format(
                    result[
                        "strain_inf_relative_to_applied_magnitude"
                    ]
                )
            )
            print(
                "  source C stress rel.err = {:.3e}".format(
                    result[
                        "source_stress_inf_relative_error"
                    ]
                )
            )
            print(
                "  deck C stress rel.err   = {:.3e}".format(
                    result[
                        "deck_stress_inf_relative_error"
                    ]
                )
            )
            print(
                "  deck/source C rel.err   = {:.3e}".format(
                    result[
                        "deck_vs_source_C_inf_relative_error"
                    ]
                )
            )

        max_strain_error = _py_builtins.max(
            result["strain_inf_relative_to_applied_magnitude"]
            for result in results
        )

        max_source_stress_error = _py_builtins.max(
            result["source_stress_inf_relative_error"]
            for result in results
        )

        max_deck_stress_error = _py_builtins.max(
            result["deck_stress_inf_relative_error"]
            for result in results
        )

        max_deck_source_C_error = _py_builtins.max(
            result["deck_vs_source_C_inf_relative_error"]
            for result in results
        )

        # The constitutive-semantics decision must use the actual constants
        # written to the Abaqus input deck. Source/deck precision loss is
        # reported separately and should be many orders smaller than the
        # structural discretization errors.
        semantic_tolerance = 1.0e-8
        serialization_tolerance = 5.0e-6

        status = (
            "PASS"
            if (
                max_strain_error <= semantic_tolerance
                and max_deck_stress_error <= semantic_tolerance
                and max_deck_source_C_error <= serialization_tolerance
            )
            else "FAIL"
        )

        summary = {
            "schema": SCHEMA,
            "status": status,
            "manifest": manifest_path,
            "case_hash": case_hash,
            "canonical_voigt_order": list(VOIGT),
            "shear_convention": "engineering",
            "abaqus_material_orientation": {
                "system": "GLOBAL",
                "additional_rotation_deg": 0.0,
            },
            "applied_strain_magnitude": strain_magnitude,
            "semantic_relative_tolerance": semantic_tolerance,
            "serialization_relative_tolerance": serialization_tolerance,
            "max_strain_inf_relative_error": max_strain_error,
            "max_source_stress_inf_relative_error": max_source_stress_error,
            "max_deck_stress_inf_relative_error": max_deck_stress_error,
            "max_deck_vs_source_C_inf_relative_error": (
                max_deck_source_C_error
            ),
            # Backward-compatible v1-v5 fields:
            "relative_tolerance": semantic_tolerance,
            "max_stress_inf_relative_error": max_source_stress_error,
            "C_hom_model_MPa": C_model,
            "abaqus_anisotropic_21_constants_MPa": abaqus_constants,
            "cases": results,
        }

        output_path = os.path.join(
            workdir,
            "constitutive_audit_summary.json",
        )

        write_json(
            output_path,
            summary,
        )

        print()
        print("-" * 88)
        print("AUDIT RESULT")
        print("-" * 88)
        print("Status               : {}".format(status))
        print(
            "max strain rel.error : {:.6e}".format(
                max_strain_error
            )
        )
        print(
            "max source-C stress error : {:.6e}".format(
                max_source_stress_error
            )
        )
        print(
            "max deck-C stress error   : {:.6e}".format(
                max_deck_stress_error
            )
        )
        print(
            "max deck/source C error   : {:.6e}".format(
                max_deck_source_C_error
            )
        )
        print(
            "semantic tolerance        : {:.1e}".format(
                semantic_tolerance
            )
        )
        print(
            "serialization tolerance   : {:.1e}".format(
                serialization_tolerance
            )
        )
        print()
        print("Wrote:")
        print("  {}".format(output_path))

        if status != "PASS":
            sys.exit(2)

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
