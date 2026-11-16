r"""Structure constants for supported finite-dimensional real algebras.

Every algebra of dimension $d$ is represented over the real basis
$\{e_0, e_1, \dots, e_{d-1}\}$ with real unit $e_0 = 1$ by a cubic
structure-constant tensor ``constants`` of shape ``(d, d, d)`` where

$$e_a e_b = \sum_{c=0}^{d-1} \text{constants}[a, b, c]\, e_c.$$

Supported Algebras
------------------
- ``"complex"`` ($d = 2$): $\mathbb{C} = \mathbb{R}[i]/(i^2 + 1)$
- ``"split_complex"`` ($d = 2$): $\mathbb{R}[j]/(j^2 - 1)$
- ``"quaternion"`` ($d = 4$): Hamilton quaternions $\mathbb{H}$
- ``"coquaternion"`` ($d = 4$): split-quaternions
- ``"cl11"`` ($d = 4$): Clifford algebra $\mathrm{Cl}(1, 1)$
- ``"octonion"`` ($d = 8$): Cayley octonions $\mathbb{O}$ (Baez/Fano orientation)

The canonical paper data pipeline remains four-dimensional
(``PAPER_COMPONENT_COUNT = 4``), while ``HyperDense`` queries
``algebra.component_count`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Kept for the four-series paper data/model API. HyperDense obtains its own
# component count from the selected algebra instead.
COMPONENT_COUNT = 4
PAPER_COMPONENT_COUNT = COMPONENT_COUNT


@dataclass(frozen=True)
class Algebra:
    r"""A named finite-dimensional real algebra equipped with a bilinear product.

    Parameters
    ----------
    name : str
        Canonical lowercase identifier of the algebra (for example,
        ``"quaternion"`` or ``"octonion"``).
    constants : torch.Tensor
        Cubic tensor of shape ``(d, d, d)`` storing structure constants
        ``constants[a, b, c]`` for $e_a e_b = \sum_c C_{abc} e_c$.

    Raises
    ------
    ValueError
        If ``constants`` is not a non-empty 3D cubic tensor ``(d, d, d)`` with
        ``d >= 1``.

    Examples
    --------
    >>> import torch
    >>> from hypercast4d.algebras import get_algebra
    >>> q = get_algebra("quaternion")
    >>> q.component_count
    4
    >>> e1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    >>> e2 = torch.tensor([0.0, 0.0, 1.0, 0.0])
    >>> q.multiply(e1, e2).tolist()  # i * j = k
    [0.0, 0.0, 0.0, 1.0]
    """

    name: str
    constants: torch.Tensor

    def __post_init__(self) -> None:
        if self.constants.ndim != 3 or len(set(self.constants.shape)) != 1:
            raise ValueError("structure constants must be a non-empty cubic tensor")
        if self.constants.shape[0] < 1:
            raise ValueError("structure constants must contain at least one component")

    @property
    def component_count(self) -> int:
        """Number of real basis components $d$ in the algebra."""
        return self.constants.shape[0]

    def multiply(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Compute the algebra product along the final tensor axis.

        Parameters
        ----------
        left : torch.Tensor
            Left operand of shape ``(..., d)`` storing basis coefficients
            ``(x_0, x_1, ..., x_{d-1})``.
        right : torch.Tensor
            Right operand of shape ``(..., d)`` broadcast-compatible with
            ``left`` along leading axes.

        Returns
        -------
        torch.Tensor
            Product tensor ``left * right`` of broadcasted shape ``(..., d)``
            with the device and dtype of ``left``.

        Raises
        ------
        ValueError
        If ``left`` or ``right`` final dimension does not equal
        ``self.component_count``.
        """
        if (
            left.shape[-1] != self.component_count
            or right.shape[-1] != self.component_count
        ):
            raise ValueError(
                "Hypercomplex operands must have final dimension "
                f"{self.component_count} for {self.name}"
            )
        constants = self.constants.to(device=left.device, dtype=left.dtype)
        return torch.einsum("...a,...b,abc->...c", left, right, constants)


def _constants(
    dimension: int,
    products: dict[tuple[int, int], tuple[float, int]],
) -> torch.Tensor:
    """Populate a ``(d, d, d)`` structure-constant table with identity ``e_0``."""
    result = torch.zeros(dimension, dimension, dimension, dtype=torch.float64)
    # Basis index 0 is always the multiplicative identity: e_0 * e_b = e_b * e_0 = e_b.
    for basis in range(dimension):
        result[0, basis, basis] = 1.0
        result[basis, 0, basis] = 1.0
    for (left, right), (sign, output) in products.items():
        result[left, right, output] = sign
    return result


def _imaginary_units(
    dimension: int, oriented_triples: tuple[tuple[int, int, int], ...]
) -> torch.Tensor:
    """Build constants for anticommuting imaginary units with square ``-1``.

    Each oriented triple ``(a, b, c)`` defines ``e_a * e_b = +e_c`` and its
    cyclic permutations, while reversing either factor negates the sign.
    """
    products: dict[tuple[int, int], tuple[float, int]] = {
        (basis, basis): (-1, 0) for basis in range(1, dimension)
    }
    for a, b, c in oriented_triples:
        for left, right, output in ((a, b, c), (b, c, a), (c, a, b)):
            products[left, right] = (1, output)
            products[right, left] = (-1, output)
    return _constants(dimension, products)


# Complex numbers R[i] / (i^2 + 1): e_1^2 = -e_0.
_COMPLEX = _constants(2, {(1, 1): (-1, 0)})

# Split-complex (hyperbolic) numbers R[j] / (j^2 - 1): e_1^2 = +e_0.
_SPLIT_COMPLEX = _constants(2, {(1, 1): (1, 0)})

# Hamilton quaternions H: i^2 = j^2 = k^2 = ijk = -1.
_QUATERNION = _constants(
    4,
    {
        (1, 1): (-1, 0),
        (1, 2): (1, 3),
        (1, 3): (-1, 2),
        (2, 1): (-1, 3),
        (2, 2): (-1, 0),
        (2, 3): (1, 1),
        (3, 1): (1, 2),
        (3, 2): (-1, 1),
        (3, 3): (-1, 0),
    },
)

# Coquaternions (split-quaternions): e_1^2 = -1, e_2^2 = +1, e_3^2 = +1.
_COQUATERNION = _constants(
    4,
    {
        (1, 1): (-1, 0),
        (1, 2): (1, 3),
        (1, 3): (-1, 2),
        (2, 1): (-1, 3),
        (2, 2): (1, 0),
        (2, 3): (-1, 1),
        (3, 1): (1, 2),
        (3, 2): (1, 1),
        (3, 3): (1, 0),
    },
)

# Real Clifford algebra Cl(1,1) matching Kycia & Niemczynowicz (2025) Table 1.
_CLIFFORD_11 = _constants(
    4,
    {
        (1, 1): (1, 0),
        (1, 2): (1, 3),
        (1, 3): (1, 2),
        (2, 1): (-1, 3),
        (2, 2): (-1, 0),
        (2, 3): (1, 1),
        (3, 1): (-1, 2),
        (3, 2): (-1, 1),
        (3, 3): (1, 0),
    },
)

# Baez/Fano-plane orientation. Octonion multiplication conventions differ by
# basis signs; pinning these 7 oriented Fano lines makes saved models reproducible.
_OCTONION = _imaginary_units(
    8,
    (
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    ),
)

ALGEBRAS: dict[str, Algebra] = {
    "complex": Algebra("complex", _COMPLEX),
    "split_complex": Algebra("split_complex", _SPLIT_COMPLEX),
    "quaternion": Algebra("quaternion", _QUATERNION),
    "coquaternion": Algebra("coquaternion", _COQUATERNION),
    "cl11": Algebra("cl11", _CLIFFORD_11),
    "octonion": Algebra("octonion", _OCTONION),
}

_ALIASES = {
    "complexes": "complex",
    "split-complex": "split_complex",
    "split complex": "split_complex",
    "quaternions": "quaternion",
    "coquaternions": "coquaternion",
    "cl(1,1)": "cl11",
    "clifford11": "cl11",
    "octonions": "octonion",
}


def get_algebra(name: str) -> Algebra:
    """Resolve an [`Algebra`][hypercast4d.algebras.Algebra] by canonical identifier or alias.

    Parameters
    ----------
    name : str
        Case-insensitive algebra name (for example ``"quaternion"``,
        ``"cl11"``, ``"Cl(1,1)"``, ``"split-complex"``, or ``"octonion"``).

    Returns
    -------
    Algebra
        Immutable algebra specification containing its structure-constant
        tensor.

    Raises
    ------
    ValueError
        If ``name`` does not match a registered algebra or alias.

    Examples
    --------
    >>> from hypercast4d.algebras import get_algebra
    >>> get_algebra("Cl(1,1)").name
    'cl11'
    >>> get_algebra("octonions").component_count
    8
    """
    key = _ALIASES.get(name.lower(), name.lower())
    try:
        return ALGEBRAS[key]
    except KeyError as error:
        choices = ", ".join(sorted(ALGEBRAS))
        raise ValueError(
            f"Unknown algebra {name!r}; choose one of {choices}"
        ) from error
