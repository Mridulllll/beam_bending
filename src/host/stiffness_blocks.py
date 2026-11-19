"""
Block decomposition of the 14x14 generalized HOST sandwich stiffness.

Generalized strain ordering:

    [ eps0      ]  0:3
    [ kappa     ]  3:6
    [ chi       ]  6:9
    [ beta      ]  9:11
    [ grad_eta  ] 11:13
    [ eta       ] 13:14

where

    eps0      : membrane strains
    kappa     : bending curvatures
    chi       : higher-order in-plane warping strains
    beta      : transverse-shear generalized variables
    grad_eta  : in-plane gradient of transverse-normal variable
    eta       : transverse-normal generalized variable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


BLOCK_RANGES = {
    "eps0": (0, 3),
    "kappa": (3, 6),
    "chi": (6, 9),
    "beta": (9, 11),
    "grad_eta": (11, 13),
    "eta": (13, 14),
}


@dataclass(frozen=True)
class GeneralizedStiffnessBlocks:
    """
    Named block representation of the generalized stiffness matrix.
    """

    full: np.ndarray
    blocks: Dict[Tuple[str, str], np.ndarray]

    def get(
        self,
        row: str,
        col: str,
    ) -> np.ndarray:
        return self.blocks[(row, col)]

    @property
    def A(self) -> np.ndarray:
        """Classical membrane stiffness."""
        return self.get("eps0", "eps0")

    @property
    def B(self) -> np.ndarray:
        """Classical membrane-bending coupling stiffness."""
        return self.get("eps0", "kappa")

    @property
    def D(self) -> np.ndarray:
        """Classical bending stiffness."""
        return self.get("kappa", "kappa")

    @property
    def membrane_warping(self) -> np.ndarray:
        return self.get("eps0", "chi")

    @property
    def bending_warping(self) -> np.ndarray:
        return self.get("kappa", "chi")

    @property
    def warping_warping(self) -> np.ndarray:
        return self.get("chi", "chi")

    @property
    def shear_shear(self) -> np.ndarray:
        return self.get("beta", "beta")

    @property
    def shear_grad_normal(self) -> np.ndarray:
        return self.get("beta", "grad_eta")

    @property
    def grad_normal_grad_normal(self) -> np.ndarray:
        return self.get("grad_eta", "grad_eta")

    @property
    def normal_normal(self) -> np.ndarray:
        return self.get("eta", "eta")


def decompose_generalized_stiffness(
    K: np.ndarray,
) -> GeneralizedStiffnessBlocks:
    """
    Split the 14x14 generalized stiffness matrix into named blocks.
    """

    K = np.asarray(K, dtype=float)

    if K.shape != (14, 14):
        raise ValueError(
            "Generalized stiffness matrix must have shape (14, 14)."
        )

    blocks = {}

    for row_name, (r0, r1) in BLOCK_RANGES.items():
        for col_name, (c0, c1) in BLOCK_RANGES.items():
            blocks[(row_name, col_name)] = (
                K[r0:r1, c0:c1].copy()
            )

    return GeneralizedStiffnessBlocks(
        full=K.copy(),
        blocks=blocks,
    )


def relative_block_norm(
    block: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Frobenius norm of a block relative to a reference matrix.
    """

    block = np.asarray(block, dtype=float)
    reference = np.asarray(reference, dtype=float)

    denom = max(
        1.0,
        float(np.linalg.norm(reference, ord="fro")),
    )

    return float(
        np.linalg.norm(block, ord="fro") / denom
    )


def block_norm_table(
    blocks: GeneralizedStiffnessBlocks,
) -> Dict[Tuple[str, str], float]:
    """
    Return Frobenius norms of all generalized stiffness blocks.
    """

    result = {}

    for key, block in blocks.blocks.items():
        result[key] = float(
            np.linalg.norm(block, ord="fro")
        )

    return result