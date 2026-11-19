#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local path configuration for HOST FEM workflows.

The tracked repository contains config/paths.example.json. Each workstation
creates its own config/paths.local.json, which is excluded from Git.

Compatible with Python 3.8+ and uses only the standard library.
"""

import json
from pathlib import Path
from typing import Any, Dict


REQUIRED_KEYS = (
    "abaqus_scratch_root",
    "tpms_stiffness_database",
)


def find_repo_root(start: Any) -> Path:
    """Find the HOST repository by walking upward from *start*."""
    path = Path(start).resolve()

    if path.is_file():
        path = path.parent

    candidates = [path]
    candidates.extend(path.parents)

    for candidate in candidates:
        if (candidate / "config" / "paths.example.json").is_file():
            return candidate
        if (candidate / ".git").exists():
            return candidate

    raise RuntimeError(
        "Could not locate the HOST repository root while walking upward from:\n"
        "{}".format(path)
    )


def load_path_config(start: Any) -> Dict[str, Any]:
    """
    Load config/paths.local.json and validate the two external data roots.
    """
    repo_root = find_repo_root(start)
    local_path = repo_root / "config" / "paths.local.json"
    example_path = repo_root / "config" / "paths.example.json"

    if not local_path.is_file():
        raise RuntimeError(
            "Local HOST path configuration does not exist:\n"
            "{}\n\n"
            "Create it from:\n"
            "{}\n\n"
            "The local file is intentionally excluded from Git.".format(
                local_path,
                example_path,
            )
        )

    try:
        data = json.loads(local_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "Could not parse local HOST path configuration:\n{}\n\n{}".format(
                local_path,
                exc,
            )
        )

    missing = [
        key
        for key in REQUIRED_KEYS
        if not str(data.get(key, "")).strip()
    ]

    if missing:
        raise RuntimeError(
            "Local HOST path configuration is missing required key(s): {}"
            .format(", ".join(missing))
        )

    scratch_root = Path(data["abaqus_scratch_root"]).expanduser()
    stiffness_database = Path(
        data["tpms_stiffness_database"]
    ).expanduser()

    if not scratch_root.is_dir():
        raise RuntimeError(
            "Configured Abaqus scratch root does not exist:\n{}".format(
                scratch_root
            )
        )

    if not stiffness_database.is_file():
        raise RuntimeError(
            "Configured TPMS stiffness database does not exist:\n{}".format(
                stiffness_database
            )
        )

    return {
        "repo_root": repo_root,
        "config_path": local_path,
        "abaqus_scratch_root": scratch_root,
        "tpms_stiffness_database": stiffness_database,
    }
