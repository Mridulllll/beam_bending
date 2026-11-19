# -*- coding: utf-8 -*-
"""
Standardized postprocessor for HOST homogenized four-point-bending Abaqus cases.

Run with the Abaqus Python interpreter, for example:

    abaqus python fem/abaqus/postprocessing/postprocess_4pb_homogenized.py ^
        --case "C:/AbaqusScratch/HOST/4pb_homogenized/<case-directory>"

The script reads the case manifest and the final frame of the requested Abaqus
step, then creates a deterministic `post` directory containing:

    summary.json
    equilibrium.csv
    global_response.csv
    face_response.csv
    section_resultants.csv
    through_thickness_profiles.csv
    energy_summary.csv

Design goals:
- Python 3.8 compatible.
- No NumPy/pandas dependency; intended for `abaqus python`.
- Preserve Abaqus/global X-Y-Z component conventions.
- Use integration-point IVOL weighting for stress/strain/energy reductions.
- Keep raw FE resultants and sign conventions explicit.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT


POSTPROCESSOR_SCHEMA = "host_4pb_homogenized_post_v2"
DEFAULT_EQUILIBRIUM_REL_TOL = 1.0e-6
DEFAULT_ENERGY_REL_TOL = 5.0e-5
ROUND_COORD_DIGITS = 9


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess a HOST homogenized 4PB Abaqus case into standardized "
            "CSV/JSON outputs."
        )
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Case directory containing manifest.json and the solved .odb.",
    )
    parser.add_argument(
        "--odb",
        default=None,
        help="Optional explicit ODB path. Defaults to <case>/<job_name>.odb.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit manifest path. Defaults to <case>/manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <case>/post.",
    )
    parser.add_argument(
        "--step",
        default=None,
        help="Optional Abaqus step name. Defaults to manifest step_name.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index to postprocess. Default: -1 (final frame).",
    )
    parser.add_argument(
        "--equilibrium-rel-tol",
        type=float,
        default=DEFAULT_EQUILIBRIUM_REL_TOL,
        help="Relative reaction-equilibrium QA tolerance.",
    )
    parser.add_argument(
        "--energy-rel-tol",
        type=float,
        default=DEFAULT_ENERGY_REL_TOL,
        help="Relative SENER versus 0.5*S:E energy QA tolerance.",
    )
    return parser.parse_args()


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def case_insensitive_repo_get(repository, name):
    if name in repository:
        return repository[name]

    wanted = str(name).upper()

    for key in repository.keys():
        if str(key).upper() == wanted:
            return repository[key]

    return None


def find_node_set(odb, name):
    root = odb.rootAssembly

    found = case_insensitive_repo_get(root.nodeSets, name)
    if found is not None:
        return found

    for instance in root.instances.values():
        found = case_insensitive_repo_get(instance.nodeSets, name)
        if found is not None:
            return found

    raise KeyError("Node set {!r} not found in ODB.".format(name))


def find_element_set_with_context(odb, name):
    """
    Return (odb_set, instance_name_or_none).

    Assembly-level sets carry their own instance namespace. Instance-level sets
    do not necessarily expose instanceNames, so the caller also receives the
    owning instance name explicitly for robust element classification.
    """
    root = odb.rootAssembly

    found = case_insensitive_repo_get(root.elementSets, name)
    if found is not None:
        return found, None

    for instance_name, instance in root.instances.items():
        found = case_insensitive_repo_get(instance.elementSets, name)
        if found is not None:
            return found, str(instance_name).upper()

    raise KeyError("Element set {!r} not found in ODB.".format(name))


def _flatten_mesh_members(member):
    if member is None:
        return

    try:
        length = len(member)
    except TypeError:
        length = None

    if length == 0:
        return

    if length is not None:
        first = member[0]

        if hasattr(first, "label"):
            for item in member:
                yield item
            return

        for sub in member:
            for item in sub:
                yield item
        return

    for item in member:
        yield item


def mesh_member_keys_from_set(odb_set, owner_instance_name=None):
    keys = set()

    instance_names = getattr(odb_set, "instanceNames", None)
    elements = getattr(odb_set, "elements", None)

    if elements is None:
        return keys

    if instance_names and len(instance_names) > 1:
        for instance_name, sub in zip(instance_names, elements):
            for item in sub:
                keys.add((str(instance_name).upper(), int(item.label)))
        return keys

    for item in _flatten_mesh_members(elements):
        instance_name = None

        if hasattr(item, "instanceName"):
            instance_name = item.instanceName

        if not instance_name and instance_names:
            instance_name = instance_names[0]

        if not instance_name and owner_instance_name:
            instance_name = owner_instance_name

        keys.add(
            (
                str(instance_name or "").upper(),
                int(item.label),
            )
        )

    return keys


def value_instance_name(value):
    instance = getattr(value, "instance", None)

    if instance is not None:
        return str(instance.name).upper()

    name = getattr(value, "instanceName", None)
    return str(name or "").upper()


def node_value_key(value):
    return (
        value_instance_name(value),
        int(value.nodeLabel),
    )


def integration_value_key(value):
    section_point = getattr(value, "sectionPoint", None)
    section_number = None

    if section_point is not None:
        section_number = getattr(section_point, "number", None)

    return (
        value_instance_name(value),
        int(value.elementLabel),
        int(getattr(value, "integrationPoint", 0) or 0),
        int(section_number or 0),
    )


def vector3(data):
    values = list(data)

    while len(values) < 3:
        values.append(0.0)

    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )


def tensor_map(field_output, data):
    labels = [
        str(item).upper()
        for item in field_output.componentLabels
    ]
    values = list(data)

    result = {}

    for index, label in enumerate(labels):
        if index < len(values):
            result[label] = float(values[index])

    return result


def scalar_value(data):
    try:
        return float(data)
    except (TypeError, ValueError):
        values = list(data)

        if len(values) != 1:
            raise ValueError(
                "Expected scalar field data, got {} values.".format(
                    len(values)
                )
            )

        return float(values[0])


def write_csv(path, fieldnames, rows):
    with open(
        str(path),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def mean(values):
    values = list(values)

    if not values:
        return None

    return sum(values) / float(len(values))


def safe_relative_difference(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1.0e-30)


def rounded_coord(value):
    return round(float(value), ROUND_COORD_DIGITS)


def build_original_node_coordinates(odb):
    mapping = {}

    for instance_name, instance in odb.rootAssembly.instances.items():
        instance_key = str(instance_name).upper()

        for node in instance.nodes:
            mapping[
                (
                    instance_key,
                    int(node.label),
                )
            ] = vector3(node.coordinates)

    return mapping


def build_element_centroids(odb, node_coordinates):
    centroids = {}

    for instance_name, instance in odb.rootAssembly.instances.items():
        instance_key = str(instance_name).upper()

        for element in instance.elements:
            coords = []

            for node_label in element.connectivity:
                key = (
                    instance_key,
                    int(node_label),
                )

                if key not in node_coordinates:
                    raise KeyError(
                        "Node {} for element {} in instance {} was not found."
                        .format(
                            node_label,
                            element.label,
                            instance_name,
                        )
                    )

                coords.append(node_coordinates[key])

            inv = 1.0 / float(len(coords))

            centroids[
                (
                    instance_key,
                    int(element.label),
                )
            ] = (
                sum(item[0] for item in coords) * inv,
                sum(item[1] for item in coords) * inv,
                sum(item[2] for item in coords) * inv,
            )

    return centroids


def get_field(frame, name):
    field = case_insensitive_repo_get(
        frame.fieldOutputs,
        name,
    )

    if field is None:
        raise KeyError(
            "Required field output {!r} is absent. Available: {}"
            .format(
                name,
                ", ".join(
                    sorted(frame.fieldOutputs.keys())
                ),
            )
        )

    return field


def field_subset_values(
    field,
    region=None,
    position=None,
):
    subset = field
    kwargs = {}

    if region is not None:
        kwargs["region"] = region

    if position is not None:
        kwargs["position"] = position

    if kwargs:
        subset = field.getSubset(**kwargs)

    return subset.values


def nodal_vector_lookup(field):
    """
    Build a {(INSTANCE, node_label): (v1, v2, v3)} lookup efficiently.

    Iterating FieldValue objects one-by-one through field.values is very slow
    for large quadratic meshes because every attribute access crosses the ODB
    Python/C++ interface. Abaqus exposes the same field in bulkDataBlocks;
    consuming those contiguous arrays is substantially faster.

    Fall back to FieldValue iteration only if bulk blocks are unavailable.
    """
    result = {}

    bulk_blocks = getattr(
        field,
        "bulkDataBlocks",
        None,
    )

    if bulk_blocks is not None:
        used_bulk = False

        for block in bulk_blocks:
            node_labels = getattr(
                block,
                "nodeLabels",
                None,
            )
            data = getattr(
                block,
                "data",
                None,
            )

            if node_labels is None or data is None:
                continue

            try:
                n_labels = len(node_labels)
                n_data = len(data)
            except TypeError:
                continue

            if n_labels == 0:
                continue

            if n_labels != n_data:
                raise RuntimeError(
                    "Nodal bulk field block has {} node labels but {} data rows."
                    .format(
                        n_labels,
                        n_data,
                    )
                )

            instance = getattr(
                block,
                "instance",
                None,
            )

            instance_name = (
                str(instance.name).upper()
                if instance is not None
                else ""
            )

            for node_label, row in zip(
                node_labels,
                data,
            ):
                result[
                    (
                        instance_name,
                        int(node_label),
                    )
                ] = vector3(
                    row
                )

            used_bulk = True

        if used_bulk:
            return result

    # Compatibility fallback for ODBs/API variants where bulk node blocks are
    # unavailable.
    for value in field.values:
        if not hasattr(
            value,
            "nodeLabel",
        ):
            continue

        result[
            node_value_key(
                value
            )
        ] = vector3(
            value.data
        )

    return result


def nodal_vector_values_for_set(field, odb_set):
    result = []

    for value in field_subset_values(
        field,
        region=odb_set,
    ):
        if not hasattr(value, "nodeLabel"):
            continue

        result.append(
            (
                node_value_key(value),
                vector3(value.data),
            )
        )

    return result


def reaction_row(
    name,
    rf_field,
    odb_set,
):
    values = nodal_vector_values_for_set(
        rf_field,
        odb_set,
    )

    return {
        "region": name,
        "node_count": len(values),
        "RF1_N": sum(
            item[1][0]
            for item in values
        ),
        "RF2_N": sum(
            item[1][1]
            for item in values
        ),
        "RF3_N": sum(
            item[1][2]
            for item in values
        ),
    }


def external_work_from_manifest(
    odb,
    u_field,
    manifest,
):
    fixture_block = manifest["fixture"]
    fixture = fixture_block["details"]

    fixture_mode = str(
        fixture_block.get(
            "mode",
            fixture.get("mode", ""),
        )
    ).strip().lower()

    if fixture_mode == "section_distributing":
        definitions = [
            (
                "left_reference_point",
                str(
                    fixture["left_load"]["set_name"]
                ),
                float(
                    fixture["left_load"]["cf3_N"]
                ),
            ),
        ]

        if fixture.get("right_load") is not None:
            definitions.append(
                (
                    "right_reference_point",
                    str(
                        fixture["right_load"]["set_name"]
                    ),
                    float(
                        fixture["right_load"]["cf3_N"]
                    ),
                )
            )

        contributions = []
        sum_force_dot_u = 0.0

        for (
            name,
            set_name,
            force_N,
        ) in definitions:
            region = find_node_set(
                odb,
                set_name,
            )

            values = nodal_vector_values_for_set(
                u_field,
                region,
            )

            if len(values) != 1:
                raise RuntimeError(
                    "Expected exactly one displacement value in {}, got {}."
                    .format(
                        set_name,
                        len(values),
                    )
                )

            u3 = float(
                values[0][1][2]
            )

            force_dot_u = (
                force_N
                * u3
            )

            contributions.append(
                {
                    "name": name,
                    "set_name": set_name,
                    "node_count": 1,
                    "force_per_node_N": force_N,
                    "sum_U3_mm": u3,
                    "force_dot_u_Nmm": force_dot_u,
                    "linear_ramp_work_Nmm": (
                        0.5 * force_dot_u
                    ),
                }
            )

            sum_force_dot_u += force_dot_u

        return {
            "contributions": contributions,
            "force_dot_u_Nmm": sum_force_dot_u,
            "linear_ramp_external_work_Nmm": (
                0.5
                * sum_force_dot_u
            ),
            "fixture_mode": fixture_mode,
        }

    definitions = [
        (
            "left_ends",
            "SET_LOAD_LEFT_LINE_ENDS",
            float(
                fixture["left_load"][
                    "endpoint_force_per_node_N"
                ]
            ),
        ),
        (
            "left_interior",
            "SET_LOAD_LEFT_LINE_INTERIOR",
            float(
                fixture["left_load"][
                    "interior_force_per_node_N"
                ]
            ),
        ),
        (
            "right_ends",
            "SET_LOAD_RIGHT_LINE_ENDS",
            float(
                fixture["right_load"][
                    "endpoint_force_per_node_N"
                ]
            ),
        ),
        (
            "right_interior",
            "SET_LOAD_RIGHT_LINE_INTERIOR",
            float(
                fixture["right_load"][
                    "interior_force_per_node_N"
                ]
            ),
        ),
    ]

    contributions = []
    sum_force_dot_u = 0.0

    for (
        name,
        set_name,
        force_per_node,
    ) in definitions:
        region = find_node_set(
            odb,
            set_name,
        )

        values = nodal_vector_values_for_set(
            u_field,
            region,
        )

        sum_u3 = sum(
            item[1][2]
            for item in values
        )

        force_dot_u = (
            force_per_node
            * sum_u3
        )

        contributions.append(
            {
                "name": name,
                "set_name": set_name,
                "node_count": len(values),
                "force_per_node_N": force_per_node,
                "sum_U3_mm": sum_u3,
                "force_dot_u_Nmm": force_dot_u,
                "linear_ramp_work_Nmm": (
                    0.5 * force_dot_u
                ),
            }
        )

        sum_force_dot_u += force_dot_u

    return {
        "contributions": contributions,
        "force_dot_u_Nmm": sum_force_dot_u,
        "linear_ramp_external_work_Nmm": (
            0.5 * sum_force_dot_u
        ),
        "fixture_mode": fixture_mode,
    }


def line_average_response_rows(
    manifest,
    u_lookup,
    node_coordinates,
):
    geometry = manifest["geometry"]

    surfaces = {
        "top": float(geometry["z_top"]),
        "midplane": 0.0,
        "bottom": float(geometry["z_bottom"]),
    }

    tol = max(
        float(
            manifest["mesh"][
                "node_selection_tolerance_mm"
            ]
        )
        * 10.0,
        1.0e-8,
    )

    grouped = {
        name: defaultdict(list)
        for name in surfaces
    }

    for key, coords in node_coordinates.items():
        if key not in u_lookup:
            continue

        x, y, z = coords
        u1, u2, u3 = u_lookup[key]

        for (
            surface_name,
            target_z,
        ) in surfaces.items():
            if abs(z - target_z) <= tol:
                grouped[
                    surface_name
                ][rounded_coord(x)].append(
                    (
                        u1,
                        u2,
                        u3,
                        y,
                        z,
                    )
                )

    x_values = sorted(
        set().union(
            *[
                set(surface_group.keys())
                for surface_group
                in grouped.values()
            ]
        )
    )

    support_left = float(
        geometry["support_left"]
    )
    support_span = float(
        geometry["support_span"]
    )
    length = float(
        geometry["length"]
    )

    rows = []

    for x in x_values:
        row = {
            "x_mm": x,
            "xi_specimen": x / length,
            "xi_support": (
                (x - support_left)
                / support_span
            ),
        }

        for surface_name in (
            "top",
            "midplane",
            "bottom",
        ):
            data = grouped[
                surface_name
            ].get(
                x,
                [],
            )

            row[
                "{}_node_count".format(
                    surface_name
                )
            ] = len(data)

            for (
                component_index,
                component_name,
            ) in enumerate(
                (
                    "U1_mm",
                    "U2_mm",
                    "U3_mm",
                )
            ):
                row[
                    "{}_{}".format(
                        surface_name,
                        component_name,
                    )
                ] = (
                    mean(
                        item[
                            component_index
                        ]
                        for item in data
                    )
                    if data
                    else None
                )

        rows.append(row)

    return rows


def build_ip_records(
    frame,
    element_centroids,
    region_key_sets,
):
    stress_field = get_field(
        frame,
        "S",
    )
    strain_field = get_field(
        frame,
        "E",
    )
    ivol_field = get_field(
        frame,
        "IVOL",
    )
    coord_field = get_field(
        frame,
        "COORD",
    )
    sener_field = get_field(
        frame,
        "SENER",
    )

    stress_values = field_subset_values(
        stress_field,
        position=INTEGRATION_POINT,
    )
    strain_values = field_subset_values(
        strain_field,
        position=INTEGRATION_POINT,
    )
    ivol_values = field_subset_values(
        ivol_field,
        position=INTEGRATION_POINT,
    )
    coord_values = field_subset_values(
        coord_field,
        position=INTEGRATION_POINT,
    )
    sener_values = field_subset_values(
        sener_field,
        position=INTEGRATION_POINT,
    )

    strain_lookup = {
        integration_value_key(value): value
        for value in strain_values
    }
    ivol_lookup = {
        integration_value_key(value): value
        for value in ivol_values
    }
    coord_lookup = {
        integration_value_key(value): value
        for value in coord_values
    }
    sener_lookup = {
        integration_value_key(value): value
        for value in sener_values
    }

    records = []

    for stress_value in stress_values:
        key = integration_value_key(
            stress_value
        )

        for (
            field_name,
            lookup,
        ) in (
            ("E", strain_lookup),
            ("IVOL", ivol_lookup),
            ("COORD", coord_lookup),
            ("SENER", sener_lookup),
        ):
            if key not in lookup:
                raise KeyError(
                    "Missing {} value for integration-point key {}."
                    .format(
                        field_name,
                        key,
                    )
                )

        (
            instance_name,
            element_label,
            integration_point,
            _,
        ) = key

        element_key = (
            instance_name,
            element_label,
        )

        if element_key not in element_centroids:
            raise KeyError(
                "Missing element centroid for {}."
                .format(
                    element_key
                )
            )

        region_name = "UNCLASSIFIED"

        for (
            candidate_name,
            key_set,
        ) in region_key_sets.items():
            if element_key in key_set:
                region_name = candidate_name
                break

        records.append(
            {
                "instance": instance_name,
                "element_label": element_label,
                "integration_point": (
                    integration_point
                ),
                "region": region_name,
                "element_centroid": (
                    element_centroids[
                        element_key
                    ]
                ),
                "coord": vector3(
                    coord_lookup[
                        key
                    ].data
                ),
                "ivol": scalar_value(
                    ivol_lookup[
                        key
                    ].data
                ),
                "sener": scalar_value(
                    sener_lookup[
                        key
                    ].data
                ),
                "stress": tensor_map(
                    stress_field,
                    stress_value.data,
                ),
                "strain": tensor_map(
                    strain_field,
                    strain_lookup[
                        key
                    ].data,
                ),
            }
        )

    return records


def get_component(mapping, name):
    return float(
        mapping.get(
            name,
            0.0,
        )
    )


def stress_strain_work_density(record):
    s = record["stress"]
    e = record["strain"]

    normal_x = (
        0.5
        * get_component(s, "S11")
        * get_component(e, "E11")
    )
    normal_y = (
        0.5
        * get_component(s, "S22")
        * get_component(e, "E22")
    )
    normal_z = (
        0.5
        * get_component(s, "S33")
        * get_component(e, "E33")
    )

    # Abaqus reports shear strain output as engineering shear strain gamma_ij.
    # Therefore 0.5*sigma:epsilon becomes 0.5*tau_ij*gamma_ij for each shear
    # pair when expressed in terms of the reported E12/E13/E23 components.
    shear_xy = (
        0.5
        * get_component(s, "S12")
        * get_component(e, "E12")
    )
    shear_xz = (
        0.5
        * get_component(s, "S13")
        * get_component(e, "E13")
    )
    shear_yz = (
        0.5
        * get_component(s, "S23")
        * get_component(e, "E23")
    )

    return {
        "normal_x": normal_x,
        "normal_y": normal_y,
        "normal_z": normal_z,
        "shear_xy": shear_xy,
        "shear_xz": shear_xz,
        "shear_yz": shear_yz,
        "total": (
            normal_x
            + normal_y
            + normal_z
            + shear_xy
            + shear_xz
            + shear_yz
        ),
    }


def energy_rows(ip_records):
    grouped = defaultdict(list)

    for record in ip_records:
        grouped[
            record["region"]
        ].append(record)

    grouped["ALL"] = list(
        ip_records
    )

    rows = []

    for region_name in sorted(
        grouped.keys()
    ):
        records = grouped[
            region_name
        ]

        volume = sum(
            record["ivol"]
            for record in records
        )

        sener_energy = sum(
            record["sener"]
            * record["ivol"]
            for record in records
        )

        component_totals = defaultdict(
            float
        )

        for record in records:
            work = (
                stress_strain_work_density(
                    record
                )
            )
            ivol = record["ivol"]

            for (
                key,
                value,
            ) in work.items():
                component_totals[
                    key
                ] += (
                    value
                    * ivol
                )

        tensor_energy = (
            component_totals[
                "total"
            ]
        )

        rows.append(
            {
                "region": region_name,
                "integration_point_count": (
                    len(records)
                ),
                "volume_mm3": volume,
                "SENER_integral_Nmm": (
                    sener_energy
                ),
                "half_sigma_colon_epsilon_Nmm": (
                    tensor_energy
                ),
                "relative_energy_difference": (
                    safe_relative_difference(
                        tensor_energy,
                        sener_energy,
                    )
                ),
                "Ux_work_Nmm": (
                    component_totals[
                        "normal_x"
                    ]
                ),
                "Uy_work_Nmm": (
                    component_totals[
                        "normal_y"
                    ]
                ),
                "Uz_work_Nmm": (
                    component_totals[
                        "normal_z"
                    ]
                ),
                "Uxy_work_Nmm": (
                    component_totals[
                        "shear_xy"
                    ]
                ),
                "Uxz_work_Nmm": (
                    component_totals[
                        "shear_xz"
                    ]
                ),
                "Uyz_work_Nmm": (
                    component_totals[
                        "shear_yz"
                    ]
                ),
            }
        )

    return rows


def computational_domain(manifest):
    geometry = manifest["geometry"]

    result = {
        "mode": "full",
        "x_min_mm": 0.0,
        "x_max_mm": float(geometry["length"]),
        "model_volume_fraction_of_full": 1.0,
        "applied_load_fraction_of_full": 1.0,
        "energy_reconstruction_factor": 1.0,
        "section_resultant_reconstruction_factor": 1.0,
    }
    result.update(
        manifest.get("domain", {})
    )
    return result


def x_slab_index(
    x_centroid,
    manifest,
    nx,
):
    domain = computational_domain(manifest)
    x_min = float(domain["x_min_mm"])
    x_max = float(domain["x_max_mm"])
    dx = (x_max - x_min) / float(nx)

    index = int(
        math.floor(
            (float(x_centroid) - x_min)
            / dx
            + 1.0e-10
        )
    )

    if index < 0:
        index = 0

    if index >= nx:
        index = nx - 1

    return index, dx


def section_resultant_rows(
    manifest,
    ip_records,
):
    geometry = manifest[
        "geometry"
    ]
    nx = int(
        manifest["mesh"][
            "derived_counts"
        ]["nx"]
    )

    domain = computational_domain(manifest)
    x_min = float(domain["x_min_mm"])
    length = (
        float(domain["x_max_mm"])
        - x_min
    )

    support_left = float(
        geometry["support_left"]
    )
    support_span = float(
        geometry["support_span"]
    )

    grouped = defaultdict(
        list
    )

    for record in ip_records:
        index, _ = x_slab_index(
            record[
                "element_centroid"
            ][0],
            manifest,
            nx,
        )

        grouped[
            index
        ].append(
            record
        )

    rows = []

    for index in range(nx):
        records = grouped.get(
            index,
            [],
        )

        dx = (
            length
            / float(nx)
        )
        x_center = (
            x_min
            + (index + 0.5) * dx
        )

        volume = sum(
            record["ivol"]
            for record in records
        )

        nx_raw = 0.0
        qz_raw = 0.0
        qy_raw = 0.0
        m_sigma_xx = 0.0

        for record in records:
            ivol = record["ivol"]
            z_ip = record[
                "coord"
            ][2]
            stress = record[
                "stress"
            ]

            nx_raw += (
                get_component(
                    stress,
                    "S11",
                )
                * ivol
                / dx
            )

            qz_raw += (
                get_component(
                    stress,
                    "S13",
                )
                * ivol
                / dx
            )

            qy_raw += (
                get_component(
                    stress,
                    "S12",
                )
                * ivol
                / dx
            )

            m_sigma_xx += (
                z_ip
                * get_component(
                    stress,
                    "S11",
                )
                * ivol
                / dx
            )

        rows.append(
            {
                "x_slab_index": index,
                "x_center_mm": x_center,
                "x_left_mm": (
                    index
                    * dx
                ),
                "x_right_mm": (
                    (index + 1)
                    * dx
                ),
                "xi_specimen": (
                    x_center
                    / length
                ),
                "xi_support": (
                    (x_center - support_left)
                    / support_span
                ),
                "dx_mm": dx,
                "integration_point_count": (
                    len(records)
                ),
                "slab_volume_mm3": volume,
                "Nx_raw_N": nx_raw,
                "Qz_raw_from_S13_N": (
                    qz_raw
                ),
                "Qy_raw_from_S12_N": (
                    qy_raw
                ),
                "int_z_sigma_xx_dA_Nmm": (
                    m_sigma_xx
                ),
                "My_beam_convention_Nmm": (
                    -m_sigma_xx
                ),
            }
        )

    return rows


def aggregate_tensor_records(records):
    total_volume = sum(
        record["ivol"]
        for record in records
    )

    if total_volume <= 0.0:
        raise ValueError(
            "Cannot aggregate records with zero volume."
        )

    unique_elements = set(
        (
            record["instance"],
            record["element_label"],
        )
        for record in records
    )

    result = {
        "integration_point_count": (
            len(records)
        ),
        "element_count": (
            len(unique_elements)
        ),
        "volume_mm3": total_volume,
    }

    for component in (
        "S11",
        "S22",
        "S33",
        "S12",
        "S13",
        "S23",
    ):
        result[
            "{}_MPa".format(
                component
            )
        ] = (
            sum(
                get_component(
                    record[
                        "stress"
                    ],
                    component,
                )
                * record[
                    "ivol"
                ]
                for record
                in records
            )
            / total_volume
        )

    for component in (
        "E11",
        "E22",
        "E33",
        "E12",
        "E13",
        "E23",
    ):
        result[
            component
        ] = (
            sum(
                get_component(
                    record[
                        "strain"
                    ],
                    component,
                )
                * record[
                    "ivol"
                ]
                for record
                in records
            )
            / total_volume
        )

    # Abaqus field output E12/E13/E23 is already engineering shear strain.
    result[
        "gamma12_engineering"
    ] = result["E12"]
    result[
        "gamma13_engineering"
    ] = result["E13"]
    result[
        "gamma23_engineering"
    ] = result["E23"]

    return result


def face_response_rows(
    manifest,
    ip_records,
):
    geometry = manifest[
        "geometry"
    ]
    nx = int(
        manifest["mesh"][
            "derived_counts"
        ]["nx"]
    )

    grouped = defaultdict(
        list
    )

    for record in ip_records:
        if record[
            "region"
        ] not in (
            "FACE_TOP",
            "FACE_BOTTOM",
        ):
            continue

        index, _ = x_slab_index(
            record[
                "element_centroid"
            ][0],
            manifest,
            nx,
        )

        z_layer = rounded_coord(
            record[
                "element_centroid"
            ][2]
        )

        grouped[
            (
                record["region"],
                index,
                z_layer,
            )
        ].append(
            record
        )

    rows = []

    for key in sorted(
        grouped.keys()
    ):
        (
            region,
            index,
            z_layer,
        ) = key

        records = grouped[
            key
        ]

        domain = computational_domain(manifest)
        x_min = float(domain["x_min_mm"])
        dx = (
            (
                float(domain["x_max_mm"])
                - x_min
            )
            / float(nx)
        )

        aggregate = (
            aggregate_tensor_records(
                records
            )
        )

        row = {
            "face": region,
            "x_slab_index": index,
            "x_center_mm": (
                x_min + (index + 0.5) * dx
            ),
            "z_element_centroid_mm": (
                z_layer
            ),
        }

        row.update(
            aggregate
        )
        rows.append(
            row
        )

    return rows


def station_profile_rows(
    manifest,
    ip_records,
):
    stations = manifest.get(
        "recommended_profile_stations",
        [],
    )

    geometry = manifest[
        "geometry"
    ]

    nx = int(
        manifest["mesh"][
            "derived_counts"
        ]["nx"]
    )

    domain = computational_domain(manifest)
    x_min = float(domain["x_min_mm"])
    dx = (
        (
            float(domain["x_max_mm"])
            - x_min
        )
        / float(nx)
    )

    slab_centers = [
        x_min
        + (index + 0.5)
        * dx
        for index
        in range(nx)
    ]

    rows = []

    for station in stations:
        name = station[
            "name"
        ]
        x_requested = float(
            station[
                "x_mm"
            ]
        )

        distances = [
            abs(
                center
                - x_requested
            )
            for center
            in slab_centers
        ]

        min_distance = min(
            distances
        )

        selected_indices = [
            index
            for (
                index,
                distance,
            ) in enumerate(
                distances
            )
            if abs(
                distance
                - min_distance
            )
            <= 1.0e-9
        ]

        selected_records = []

        for record in ip_records:
            index, _ = x_slab_index(
                record[
                    "element_centroid"
                ][0],
                manifest,
                nx,
            )

            if index in selected_indices:
                selected_records.append(
                    record
                )

        grouped = defaultdict(
            list
        )

        for record in selected_records:
            # Preserve the actual integration-point through-thickness
            # coordinate. Grouping by element-centroid z (the former v1
            # behavior) erased the multiple Gauss-point levels of C3D20R and
            # made local face comparisons dependent on layer thickness.
            grouped[
                (
                    record["region"],
                    rounded_coord(
                        record[
                            "coord"
                        ][2]
                    ),
                )
            ].append(
                record
            )

        sampled_centers = [
            slab_centers[
                index
            ]
            for index
            in selected_indices
        ]

        for key in sorted(
            grouped.keys(),
            key=lambda item: item[1],
        ):
            (
                region,
                z_ip,
            ) = key

            grouped_records = grouped[
                key
            ]

            aggregate = (
                aggregate_tensor_records(
                    grouped_records
                )
            )

            total_volume = sum(
                record["ivol"]
                for record in grouped_records
            )

            z_element_centroid_mean = (
                sum(
                    record[
                        "element_centroid"
                    ][2]
                    * record["ivol"]
                    for record
                    in grouped_records
                )
                / total_volume
            )

            row = {
                "station": name,
                "purpose": station.get(
                    "purpose",
                    "",
                ),
                "x_requested_mm": (
                    x_requested
                ),
                "x_sampled_mean_mm": (
                    mean(
                        sampled_centers
                    )
                ),
                "x_sampled_min_mm": (
                    min(
                        sampled_centers
                    )
                ),
                "x_sampled_max_mm": (
                    max(
                        sampled_centers
                    )
                ),
                "x_slab_count": (
                    len(
                        selected_indices
                    )
                ),
                "z_ip_mm": (
                    z_ip
                ),
                # Retained for backward compatibility and diagnostics. The
                # actual profile coordinate is z_ip_mm.
                "z_element_centroid_mm": (
                    z_element_centroid_mean
                ),
                "region": region,
            }

            row.update(
                aggregate
            )
            rows.append(
                row
            )

    return rows


def midspan_summary(
    global_rows,
    manifest,
):
    target = float(
        manifest["geometry"][
            "midspan"
        ]
    )

    if not global_rows:
        return {}

    row = min(
        global_rows,
        key=lambda item: abs(
            float(
                item["x_mm"]
            )
            - target
        ),
    )

    top = safe_float(
        row.get(
            "top_U3_mm"
        )
    )
    bottom = safe_float(
        row.get(
            "bottom_U3_mm"
        )
    )
    mid = safe_float(
        row.get(
            "midplane_U3_mm"
        )
    )

    if (
        mid is None
        and top is not None
        and bottom is not None
    ):
        mid = (
            0.5
            * (
                top
                + bottom
            )
        )

    return {
        "x_requested_mm": target,
        "x_sampled_mm": row[
            "x_mm"
        ],
        "U3_top_avg_mm": top,
        "U3_midplane_avg_mm": mid,
        "U3_bottom_avg_mm": bottom,
        "top_minus_bottom_U3_mm": (
            top - bottom
            if (
                top is not None
                and bottom is not None
            )
            else None
        ),
    }


def parse_sta_completed(
    case_dir,
    job_name,
):
    path = (
        case_dir
        / "{}.sta".format(
            job_name
        )
    )

    if not path.exists():
        return {
            "sta_path": str(
                path
            ),
            "exists": False,
            "completed_successfully": (
                False
            ),
        }

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    completed = (
        "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
        in text.upper()
    )

    return {
        "sta_path": str(
            path
        ),
        "exists": True,
        "completed_successfully": (
            completed
        ),
    }


def ensure_required_builder_schema(
    manifest,
):
    schema = str(
        manifest.get(
            "schema",
            "",
        )
    )

    if not schema.startswith(
        "host_4pb_homogenized_beam_v"
    ):
        raise ValueError(
            "Unsupported manifest schema {!r}."
            .format(
                schema
            )
        )


def main():
    args = parse_args()

    case_dir = Path(
        args.case
    ).resolve()

    if not case_dir.is_dir():
        raise FileNotFoundError(
            "Case directory not found: {}"
            .format(
                case_dir
            )
        )

    manifest_path = (
        Path(
            args.manifest
        ).resolve()
        if args.manifest
        else (
            case_dir
            / "manifest.json"
        )
    )

    manifest = load_json(
        manifest_path
    )

    ensure_required_builder_schema(
        manifest
    )

    job_name = str(
        manifest["job_name"]
    )

    odb_path = (
        Path(
            args.odb
        ).resolve()
        if args.odb
        else (
            case_dir
            / "{}.odb".format(
                job_name
            )
        )
    )

    if not odb_path.exists():
        raise FileNotFoundError(
            "ODB not found: {}"
            .format(
                odb_path
            )
        )

    output_dir = (
        Path(
            args.output_dir
        ).resolve()
        if args.output_dir
        else (
            case_dir
            / "post"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    step_name = (
        args.step
        or str(
            manifest[
                "step_name"
            ]
        )
    )

    print(
        "=" * 88
    )
    print(
        "HOST / ABAQUS 4PB HOMOGENIZED ODB POSTPROCESSOR"
    )
    print(
        "=" * 88
    )
    print(
        "Case     : {}".format(
            case_dir
        )
    )
    print(
        "Manifest : {}".format(
            manifest_path
        )
    )
    print(
        "ODB      : {}".format(
            odb_path
        )
    )
    print(
        "Output   : {}".format(
            output_dir
        )
    )
    print(
        "Step     : {}".format(
            step_name
        )
    )
    print(
        "Frame    : {}".format(
            args.frame
        )
    )
    print()

    odb = openOdb(
        path=str(
            odb_path
        ),
        readOnly=True,
    )

    try:
        step = case_insensitive_repo_get(
            odb.steps,
            step_name,
        )

        if step is None:
            raise KeyError(
                "Step {!r} not found. Available: {}"
                .format(
                    step_name,
                    ", ".join(
                        odb.steps.keys()
                    ),
                )
            )

        frame = step.frames[
            args.frame
        ]

        u_field = get_field(
            frame,
            "U",
        )
        rf_field = get_field(
            frame,
            "RF",
        )

        print("Reading mesh coordinates...")
        node_coordinates = (
            build_original_node_coordinates(
                odb
            )
        )

        print(
            "  user mesh nodes: {}"
            .format(
                len(
                    node_coordinates
                )
            )
        )

        print("Building element centroids...")
        element_centroids = (
            build_element_centroids(
                odb,
                node_coordinates,
            )
        )

        print(
            "  elements: {}"
            .format(
                len(
                    element_centroids
                )
            )
        )

        domain = computational_domain(manifest)
        domain_mode = str(
            domain.get("mode", "full")
        ).strip().lower()

        support_left = find_node_set(
            odb,
            "SET_SUPPORT_LEFT_LINE",
        )
        left_row = reaction_row(
            "support_left",
            rf_field,
            support_left,
        )

        if domain_mode == "quarter_xy":
            right_row = None
        else:
            support_right = find_node_set(
                odb,
                "SET_SUPPORT_RIGHT_LINE",
            )
            right_row = reaction_row(
                "support_right",
                rf_field,
                support_right,
            )

        reaction_rows_present = [left_row]
        if right_row is not None:
            reaction_rows_present.append(right_row)

        total_reaction = {
            "region": "supports_total",
            "node_count": sum(
                int(row["node_count"])
                for row in reaction_rows_present
            ),
            "RF1_N": sum(
                float(row["RF1_N"])
                for row in reaction_rows_present
            ),
            "RF2_N": sum(
                float(row["RF2_N"])
                for row in reaction_rows_present
            ),
            "RF3_N": sum(
                float(row["RF3_N"])
                for row in reaction_rows_present
            ),
        }

        applied_load_magnitude = float(
            manifest["fixture"].get(
                "model_applied_load_N",
                manifest["fixture"]["total_applied_load_N"],
            )
        )

        equilibrium_residual = (
            total_reaction[
                "RF3_N"
            ]
            - applied_load_magnitude
        )

        equilibrium_relative = (
            abs(
                equilibrium_residual
            )
            / max(
                applied_load_magnitude,
                1.0e-30,
            )
        )

        equilibrium_rows = list(
            reaction_rows_present
        ) + [
            total_reaction,
            {
                "region": (
                    "expected_total_vertical_reaction"
                ),
                "node_count": "",
                "RF1_N": "",
                "RF2_N": "",
                "RF3_N": (
                    applied_load_magnitude
                ),
            },
            {
                "region": (
                    "vertical_equilibrium_residual"
                ),
                "node_count": "",
                "RF1_N": "",
                "RF2_N": "",
                "RF3_N": (
                    equilibrium_residual
                ),
            },
        ]

        print("Reading nodal displacement field...")
        u_lookup = nodal_vector_lookup(
            u_field
        )

        print(
            "  nodal U values: {}"
            .format(
                len(
                    u_lookup
                )
            )
        )

        global_rows = (
            line_average_response_rows(
                manifest,
                u_lookup,
                node_coordinates,
            )
        )

        midspan = midspan_summary(
            global_rows,
            manifest,
        )

        external_work = (
            external_work_from_manifest(
                odb,
                u_field,
                manifest,
            )
        )

        (
            core_set,
            core_owner,
        ) = find_element_set_with_context(
            odb,
            "SET_CORE",
        )
        (
            face_top_set,
            face_top_owner,
        ) = find_element_set_with_context(
            odb,
            "SET_FACE_TOP",
        )
        (
            face_bottom_set,
            face_bottom_owner,
        ) = find_element_set_with_context(
            odb,
            "SET_FACE_BOTTOM",
        )

        region_key_sets = {
            "CORE": (
                mesh_member_keys_from_set(
                    core_set,
                    core_owner,
                )
            ),
            "FACE_TOP": (
                mesh_member_keys_from_set(
                    face_top_set,
                    face_top_owner,
                )
            ),
            "FACE_BOTTOM": (
                mesh_member_keys_from_set(
                    face_bottom_set,
                    face_bottom_owner,
                )
            ),
        }

        print("Reading integration-point fields...")
        ip_records = build_ip_records(
            frame,
            element_centroids,
            region_key_sets,
        )

        print(
            "  integration-point records: {}"
            .format(
                len(
                    ip_records
                )
            )
        )

        energy = energy_rows(
            ip_records
        )

        section_rows = (
            section_resultant_rows(
                manifest,
                ip_records,
            )
        )

        face_rows = (
            face_response_rows(
                manifest,
                ip_records,
            )
        )

        profile_rows = (
            station_profile_rows(
                manifest,
                ip_records,
            )
        )

        all_energy_row = next(
            row
            for row in energy
            if row[
                "region"
            ] == "ALL"
        )

        sener_total = float(
            all_energy_row[
                "SENER_integral_Nmm"
            ]
        )

        tensor_energy_total = float(
            all_energy_row[
                "half_sigma_colon_epsilon_Nmm"
            ]
        )

        energy_relative = float(
            all_energy_row[
                "relative_energy_difference"
            ]
        )

        external_work_value = float(
            external_work[
                "linear_ramp_external_work_Nmm"
            ]
        )

        external_internal_relative = (
            safe_relative_difference(
                external_work_value,
                sener_total,
            )
        )

        sta_status = parse_sta_completed(
            case_dir,
            job_name,
        )

        required_fields = [
            "U",
            "RF",
            "S",
            "E",
            "IVOL",
            "SENER",
            "COORD",
        ]

        field_availability = {
            name: (
                case_insensitive_repo_get(
                    frame.fieldOutputs,
                    name,
                )
                is not None
            )
            for name
            in required_fields
        }

        qa = {
            "solver_completed": {
                "status": (
                    "PASS"
                    if sta_status[
                        "completed_successfully"
                    ]
                    else "FAIL"
                ),
                "completed_successfully": (
                    sta_status[
                        "completed_successfully"
                    ]
                ),
            },
            "required_fields": {
                "status": (
                    "PASS"
                    if all(
                        field_availability.values()
                    )
                    else "FAIL"
                ),
                "availability": (
                    field_availability
                ),
            },
            "reaction_equilibrium": {
                "status": (
                    "PASS"
                    if equilibrium_relative
                    <= float(
                        args.equilibrium_rel_tol
                    )
                    else "FAIL"
                ),
                "expected_R3_N": (
                    applied_load_magnitude
                ),
                "computed_R3_N": (
                    total_reaction[
                        "RF3_N"
                    ]
                ),
                "residual_N": (
                    equilibrium_residual
                ),
                "relative_error": (
                    equilibrium_relative
                ),
                "relative_tolerance": (
                    float(
                        args.equilibrium_rel_tol
                    )
                ),
            },
            "energy_density_consistency": {
                "status": (
                    "PASS"
                    if energy_relative
                    <= float(
                        args.energy_rel_tol
                    )
                    else "FAIL"
                ),
                "SENER_integral_Nmm": (
                    sener_total
                ),
                "half_sigma_colon_epsilon_Nmm": (
                    tensor_energy_total
                ),
                "relative_error": (
                    energy_relative
                ),
                "relative_tolerance": (
                    float(
                        args.energy_rel_tol
                    )
                ),
            },
            "external_internal_work": {
                "status": (
                    "PASS"
                    if external_internal_relative
                    <= max(
                        float(
                            args.energy_rel_tol
                        ),
                        5.0e-4,
                    )
                    else "FAIL"
                ),
                "linear_ramp_external_work_Nmm": (
                    external_work_value
                ),
                "SENER_integral_Nmm": (
                    sener_total
                ),
                "relative_error": (
                    external_internal_relative
                ),
                "note": (
                    "For this linear static ramp from zero load, external "
                    "work is 0.5*sum(F_i*U_i)."
                ),
            },
            "left_right_reaction_diagnostic": {
                "applicable": right_row is not None,
                "left_R3_N": left_row["RF3_N"],
                "right_R3_N": (
                    None
                    if right_row is None
                    else right_row["RF3_N"]
                ),
                "absolute_difference_N": (
                    None
                    if right_row is None
                    else abs(
                        float(left_row["RF3_N"])
                        - float(right_row["RF3_N"])
                    )
                ),
                "relative_difference_by_total_load": (
                    None
                    if right_row is None
                    else (
                        abs(
                            float(left_row["RF3_N"])
                            - float(right_row["RF3_N"])
                        )
                        / max(applied_load_magnitude, 1.0e-30)
                    )
                ),
                "note": (
                    "Not applicable for quarter_xy because only the retained "
                    "left support exists in the computational domain."
                    if right_row is None
                    else (
                        "Diagnostic only. Exact left/right equality is not "
                        "required when the anisotropic constitutive law lacks "
                        "reflection symmetry."
                    )
                ),
            },
        }

        overall_pass = all(
            item.get(
                "status"
            ) == "PASS"
            for item in qa.values()
            if "status" in item
        )

        equilibrium_path = (
            output_dir
            / "equilibrium.csv"
        )
        global_path = (
            output_dir
            / "global_response.csv"
        )
        face_path = (
            output_dir
            / "face_response.csv"
        )
        section_path = (
            output_dir
            / "section_resultants.csv"
        )
        profile_path = (
            output_dir
            / "through_thickness_profiles.csv"
        )
        energy_path = (
            output_dir
            / "energy_summary.csv"
        )
        summary_path = (
            output_dir
            / "summary.json"
        )

        write_csv(
            equilibrium_path,
            [
                "region",
                "node_count",
                "RF1_N",
                "RF2_N",
                "RF3_N",
            ],
            equilibrium_rows,
        )

        global_fieldnames = [
            "x_mm",
            "xi_specimen",
            "xi_support",
        ]

        for surface in (
            "top",
            "midplane",
            "bottom",
        ):
            global_fieldnames.extend(
                [
                    "{}_node_count".format(
                        surface
                    ),
                    "{}_U1_mm".format(
                        surface
                    ),
                    "{}_U2_mm".format(
                        surface
                    ),
                    "{}_U3_mm".format(
                        surface
                    ),
                ]
            )

        write_csv(
            global_path,
            global_fieldnames,
            global_rows,
        )

        tensor_aggregate_fields = [
            "integration_point_count",
            "element_count",
            "volume_mm3",
            "S11_MPa",
            "S22_MPa",
            "S33_MPa",
            "S12_MPa",
            "S13_MPa",
            "S23_MPa",
            "E11",
            "E22",
            "E33",
            "E12",
            "E13",
            "E23",
            "gamma12_engineering",
            "gamma13_engineering",
            "gamma23_engineering",
        ]

        write_csv(
            face_path,
            [
                "face",
                "x_slab_index",
                "x_center_mm",
                "z_element_centroid_mm",
            ]
            + tensor_aggregate_fields,
            face_rows,
        )

        write_csv(
            section_path,
            [
                "x_slab_index",
                "x_center_mm",
                "x_left_mm",
                "x_right_mm",
                "xi_specimen",
                "xi_support",
                "dx_mm",
                "integration_point_count",
                "slab_volume_mm3",
                "Nx_raw_N",
                "Qz_raw_from_S13_N",
                "Qy_raw_from_S12_N",
                "int_z_sigma_xx_dA_Nmm",
                "My_beam_convention_Nmm",
            ],
            section_rows,
        )

        write_csv(
            profile_path,
            [
                "station",
                "purpose",
                "x_requested_mm",
                "x_sampled_mean_mm",
                "x_sampled_min_mm",
                "x_sampled_max_mm",
                "x_slab_count",
                "z_ip_mm",
                "z_element_centroid_mm",
                "region",
            ]
            + tensor_aggregate_fields,
            profile_rows,
        )

        write_csv(
            energy_path,
            [
                "region",
                "integration_point_count",
                "volume_mm3",
                "SENER_integral_Nmm",
                "half_sigma_colon_epsilon_Nmm",
                "relative_energy_difference",
                "Ux_work_Nmm",
                "Uy_work_Nmm",
                "Uz_work_Nmm",
                "Uxy_work_Nmm",
                "Uxz_work_Nmm",
                "Uyz_work_Nmm",
            ],
            energy,
        )

        summary = {
            "schema": (
                POSTPROCESSOR_SCHEMA
            ),
            "source": {
                "case_directory": str(
                    case_dir
                ),
                "manifest_path": str(
                    manifest_path
                ),
                "manifest_sha256": (
                    sha256_file(
                        manifest_path
                    )
                ),
                "odb_path": str(
                    odb_path
                ),
                "odb_size_bytes": int(
                    odb_path.stat().st_size
                ),
                "builder_case_id": (
                    manifest["case"][
                        "case_id"
                    ]
                ),
                "builder_case_hash": (
                    manifest["case"][
                        "case_hash"
                    ]
                ),
                "builder_schema": (
                    manifest[
                        "schema"
                    ]
                ),
                "model_implementation_revision": (
                    manifest[
                        "model_implementation_revision"
                    ]
                ),
                "job_name": job_name,
                "step_name": (
                    step.name
                ),
                "frame_index": (
                    args.frame
                ),
                "frame_value": float(
                    frame.frameValue
                ),
            },
            "units": {
                "length": "mm",
                "force": "N",
                "stress": "MPa",
                "energy": "N*mm",
            },
            "sign_conventions": {
                "coordinates": (
                    "Global X=beam longitudinal, Y=width, Z=thickness."
                ),
                "transverse_displacement": (
                    "U3<0 corresponds to downward deflection."
                ),
                "section_resultants": {
                    "Nx_raw_N": (
                        "integral_A(S11)dA"
                    ),
                    "Qz_raw_from_S13_N": (
                        "integral_A(S13)dA"
                    ),
                    "Qy_raw_from_S12_N": (
                        "integral_A(S12)dA"
                    ),
                    "int_z_sigma_xx_dA_Nmm": (
                        "integral_A(z*S11)dA"
                    ),
                    "My_beam_convention_Nmm": (
                        "-integral_A(z*S11)dA; positive for conventional "
                        "sagging when top S11 is compressive."
                    ),
                },
                "shear_strain": (
                    "Abaqus reports E12/E13/E23 as engineering shear strains "
                    "gamma_ij. The gamma*_engineering columns are explicit "
                    "aliases of those reported components."
                ),
            },
            "reduction_methods": {
                "section_resultants": (
                    "Integration-point S and IVOL are integrated over each "
                    "structured x slab and divided by slab width dx, yielding "
                    "the mean cross-sectional resultant over that slab."
                ),
                "through_thickness_profiles": (
                    "Width-averaged, IVOL-weighted integration-point fields "
                    "grouped by the actual integration-point coordinate z_ip. "
                    "The mean element-centroid z is retained only as a "
                    "diagnostic/backward-compatible column. If a requested "
                    "station lies on an x nodal plane, both equidistant "
                    "adjacent x slabs are averaged."
                ),
                "face_response": (
                    "IVOL-weighted integration-point averages, separated by "
                    "face sheet, x slab, and element-centroid z layer."
                ),
                "energy": (
                    "SENER is integrated with IVOL. A separate tensor-work "
                    "check evaluates 0.5*S:E; because Abaqus reports "
                    "engineering shear strain gamma_ij, each shear contribution "
                    "is 0.5*tau_ij*gamma_ij."
                ),
            },
            "equilibrium": {
                "left_support": (
                    left_row
                ),
                "right_support": (
                    right_row
                ),
                "supports_total": (
                    total_reaction
                ),
                "expected_vertical_reaction_N": (
                    applied_load_magnitude
                ),
                "vertical_residual_N": (
                    equilibrium_residual
                ),
                "vertical_relative_error": (
                    equilibrium_relative
                ),
            },
            "domain": domain,
            "midspan": midspan,
            "external_work": (
                external_work
            ),
            "energy": {
                "SENER_integral_Nmm": (
                    sener_total
                ),
                "half_sigma_colon_epsilon_Nmm": (
                    tensor_energy_total
                ),
                "linear_ramp_external_work_Nmm": (
                    external_work_value
                ),
                "full_model_reconstructed_SENER_Nmm": (
                    sener_total
                    * float(
                        domain.get(
                            "energy_reconstruction_factor",
                            1.0,
                        )
                    )
                ),
            },
            "qa": qa,
            "overall_status": (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
            "outputs": {
                "equilibrium_csv": str(
                    equilibrium_path
                ),
                "global_response_csv": str(
                    global_path
                ),
                "face_response_csv": str(
                    face_path
                ),
                "section_resultants_csv": str(
                    section_path
                ),
                "through_thickness_profiles_csv": str(
                    profile_path
                ),
                "energy_summary_csv": str(
                    energy_path
                ),
            },
        }

        with open(
            str(
                summary_path
            ),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write(
                "\n"
            )

        print(
            "Completed."
        )
        print()
        print(
            "QA overall : {}".format(
                summary[
                    "overall_status"
                ]
            )
        )
        print(
            "Reaction R3 : {:.9g} N (expected {:.9g} N, rel.err {:.3e})"
            .format(
                total_reaction[
                    "RF3_N"
                ],
                applied_load_magnitude,
                equilibrium_relative,
            )
        )

        midspan_value = midspan.get(
            "U3_midplane_avg_mm"
        )

        if midspan_value is not None:
            print(
                "Midspan U3  : {:.9g} mm"
                .format(
                    float(
                        midspan_value
                    )
                )
            )

        print(
            "SENER       : {:.9g} N*mm"
            .format(
                sener_total
            )
        )
        print(
            "0.5*S:E     : {:.9g} N*mm (rel.err {:.3e})"
            .format(
                tensor_energy_total,
                energy_relative,
            )
        )
        print(
            "0.5*F.U     : {:.9g} N*mm (vs SENER rel.err {:.3e})"
            .format(
                external_work_value,
                external_internal_relative,
            )
        )
        print()
        print(
            "Wrote:"
        )

        for path in (
            summary_path,
            equilibrium_path,
            global_path,
            face_path,
            section_path,
            profile_path,
            energy_path,
        ):
            print(
                "  {}".format(
                    path
                )
            )

    finally:
        odb.close()


if __name__ == "__main__":
    main()
