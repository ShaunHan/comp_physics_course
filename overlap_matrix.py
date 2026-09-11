"""Overlap-matrix (OM) construction for atomistic structural fingerprints.

The implementation follows the main ideas used by overlap-matrix fingerprint
methods:

1. place simple Gaussian ``s`` and ``p`` functions on atoms;
2. use widths related to atomic covalent radii;
3. build and normalize the overlap matrix;
4. for atom-centred matrices, apply a smooth cutoff so neighbours disappear
   continuously at the cutoff radius.

The public functions return overlap matrices, not their eigenvalues.  The
rotation/translation-invariant fingerprint is obtained by diagonalizing each
symmetric overlap matrix, for example with ``numpy.linalg.eigvalsh``.

References
----------
- Sadeghi et al., J. Chem. Phys. 139, 184118 (2013), overlap-matrix
  fingerprints for structures.
- Zhu et al., J. Chem. Phys. 144, 034203 (2016), local/crystalline OM
  fingerprints.
- Minima Hopping OMFP implementation:
  https://gitlab.com/goedeckergroup/ase_mh/-/tree/main/minimahopping/omfp

Notes
-----
This compact implementation keeps one Gaussian radial width per atom and uses
an ``s`` shell or a complete ``sp`` shell.  It is intended for small examples
and transparent inspection rather than as a drop-in replacement for an
optimized production OMFP implementation.
"""

from __future__ import annotations

import numpy as np


_ORBITAL_AXES = {
    "s": None,
    "px": 0,
    "py": 1,
    "pz": 2,
}


def _covalent_radii(numbers: np.ndarray) -> np.ndarray:
    """Return ASE covalent radii (Å) for atomic numbers."""
    try:
        from ase.data import covalent_radii
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "ASE is required. Install it with `pip install ase` before using "
            "overlap_matrix.py."
        ) from exc

    radii = np.asarray(covalent_radii, dtype=float)[numbers]
    if np.any(radii <= 0):
        bad = numbers[radii <= 0]
        raise ValueError(f"No positive covalent radius available for Z={bad.tolist()}")
    return radii


def _smooth_cutoff(distance: np.ndarray, cutoff: float, power: int = 2) -> np.ndarray:
    """Polynomial cutoff: [1 - (r/rc)^2]^power for r <= rc, otherwise 0."""
    distance = np.asarray(distance, dtype=float)
    x = 1.0 - (distance / cutoff) ** 2
    return np.where(distance <= cutoff, np.clip(x, 0.0, None) ** power, 0.0)


def _orbital_labels(orbitals: str) -> tuple[str, ...]:
    orbitals = orbitals.lower().replace("+", "")
    if orbitals == "s":
        return ("s",)
    if orbitals == "sp":
        return ("s", "px", "py", "pz")
    raise ValueError("orbitals must be 's' or 'sp'")


def _raw_overlap(
    center_a: np.ndarray,
    alpha: float,
    label_a: str,
    center_b: np.ndarray,
    beta: float,
    label_b: str,
) -> float:
    """Unnormalized overlap integral for Cartesian s/p Gaussian functions."""
    p = alpha + beta
    mu = alpha * beta / p
    displacement = center_b - center_a

    ss = (np.pi / p) ** 1.5 * np.exp(-mu * np.dot(displacement, displacement))

    axis_a = _ORBITAL_AXES[label_a]
    axis_b = _ORBITAL_AXES[label_b]

    if axis_a is None and axis_b is None:
        return float(ss)

    # Gaussian product centre P gives P-A and P-B below.
    pa = beta * displacement / p
    pb = -alpha * displacement / p

    if axis_a is not None and axis_b is None:
        return float(pa[axis_a] * ss)
    if axis_a is None and axis_b is not None:
        return float(pb[axis_b] * ss)

    correction = 1.0 / (2.0 * p) if axis_a == axis_b else 0.0
    return float((pa[axis_a] * pb[axis_b] + correction) * ss)


def _overlap_matrix_from_arrays(
    positions: np.ndarray,
    numbers: np.ndarray,
    *,
    orbitals: str = "sp",
    width_scale: float = 1.0,
    atom_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Build and return a normalized symmetric overlap matrix."""
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    if positions.shape != (len(numbers), 3):
        raise ValueError("positions must have shape (N, 3)")
    if width_scale <= 0:
        raise ValueError("width_scale must be positive")

    labels = _orbital_labels(orbitals)
    radii = _covalent_radii(numbers) * float(width_scale)

    if atom_weights is None:
        atom_weights = np.ones(len(numbers), dtype=float)
    else:
        atom_weights = np.asarray(atom_weights, dtype=float)
        if atom_weights.shape != (len(numbers),):
            raise ValueError("atom_weights must have shape (N,)")

    centers: list[np.ndarray] = []
    alphas: list[float] = []
    orbital_labels: list[str] = []
    orbital_weights: list[float] = []

    for position, radius, weight in zip(positions, radii, atom_weights):
        alpha = 1.0 / (2.0 * radius**2)
        for label in labels:
            centers.append(position)
            alphas.append(alpha)
            orbital_labels.append(label)
            orbital_weights.append(weight)

    centers = np.asarray(centers, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    orbital_weights = np.asarray(orbital_weights, dtype=float)
    n_orbitals = len(orbital_labels)

    overlap = np.empty((n_orbitals, n_orbitals), dtype=float)
    for i in range(n_orbitals):
        for j in range(i, n_orbitals):
            value = _raw_overlap(
                centers[i], alphas[i], orbital_labels[i],
                centers[j], alphas[j], orbital_labels[j],
            )
            overlap[i, j] = value
            overlap[j, i] = value

    # Normalize each orbital to unit self-overlap and apply the atom-centred
    # cutoff amplitudes through an outer product.
    diagonal = np.diag(overlap)
    if np.any(diagonal <= 0):
        raise FloatingPointError("Non-positive self-overlap encountered")

    normalization = orbital_weights / np.sqrt(diagonal)
    overlap *= np.outer(normalization, normalization)

    # Remove round-off asymmetry so the result is explicitly symmetric.
    return 0.5 * (overlap + overlap.T)


def global_overlap_matrix(
    atoms,
    *,
    orbitals: str = "sp",
    width_scale: float = 1.0,
) -> np.ndarray:
    """Return the normalized global overlap matrix for an ASE ``Atoms`` object."""
    return _overlap_matrix_from_arrays(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        orbitals=orbitals,
        width_scale=width_scale,
    )


def local_overlap_matrices(
    atoms,
    *,
    cutoff: float = 3.0,
    orbitals: str = "sp",
    width_scale: float = 1.0,
    cutoff_power: int = 2,
) -> list[np.ndarray]:
    """Return one atom-centred overlap matrix for every atom.

    The neighbourhood is centred on each atom.  For periodic systems, ASE's
    minimum-image vectors are used.  Orbitals are smoothly down-weighted as
    their atoms approach ``cutoff``.

    Returns
    -------
    list of numpy.ndarray
        ``matrices[i]`` is the symmetric local overlap matrix centred on atom
        ``i``.  Matrix sizes can differ when different atoms have different
        numbers of neighbours.  For a small CH4 molecule with ``cutoff=3.0`` Å,
        all five matrices have the same size and can be stacked with
        ``np.array(matrices)`` before calling ``np.linalg.eigvalsh``.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if cutoff_power < 1:
        raise ValueError("cutoff_power must be >= 1")

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    indices = np.arange(len(atoms))
    matrices: list[np.ndarray] = []

    for centre in range(len(atoms)):
        if np.any(atoms.get_pbc()):
            vectors = atoms.get_distances(centre, indices, mic=True, vector=True)
        else:
            vectors = positions - positions[centre]

        distances = np.linalg.norm(vectors, axis=1)
        keep = distances <= cutoff

        local_positions = vectors[keep]
        local_numbers = numbers[keep]
        weights = _smooth_cutoff(distances[keep], cutoff, power=cutoff_power)

        matrices.append(
            _overlap_matrix_from_arrays(
                local_positions,
                local_numbers,
                orbitals=orbitals,
                width_scale=width_scale,
                atom_weights=weights,
            )
        )

    return matrices


__all__ = ["global_overlap_matrix", "local_overlap_matrices"]
