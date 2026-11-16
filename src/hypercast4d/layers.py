"""PyTorch layers backed by explicit hypercomplex structure constants.

This module provides two hypercomplex linear layers:

- [`HyperDense`][hypercast4d.layers.HyperDense] — strict hypercomplex linear
  projection requiring the input's last dimension to equal ``d * in_features``
  and producing ``d * out_features`` features in component-major order.
- [`ShapePreservingHyperDense`][hypercast4d.layers.ShapePreservingHyperDense] —
  drop-in adapter that zero-pads arbitrary last-axis feature widths ``C`` up to
  ``ceil(C / d) * d``, applies a square ``HyperDense`` map, and crops back to
  ``C`` while guaranteeing memory contiguity for downstream ``view()`` calls.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .algebras import Algebra, get_algebra


class HyperDense(nn.Module):
    r"""Hypercomplex dense layer using input-by-weight bilinear products.

    Inputs and outputs use **component-major layout**: all ``in_features``
    units of component $e_0$ come first, followed by ``in_features`` units of
    $e_1$, through $e_{d-1}$. Let $x \in \mathbb{R}^{\dots \times d \times n}$
    be the reshaped input and $W \in \mathbb{R}^{d \times n \times m}$ the
    component weight kernels. Using the structure constants $C_{bac}$ for
    $e_b e_a$, output component $c$ is computed as

    $$y_{\dots, c, o} = \sum_{b=0}^{d-1} \sum_{a=0}^{d-1} \sum_{i=0}^{n-1}
        C_{b, a, c}\, x_{\dots, b, i}\, W_{a, i, o} + b_{c, o}.$$

    For an algebra of dimension $d$, this learns $d n m$ kernel weights instead
    of the $d^2 n m$ weights of an unconstrained real ``nn.Linear(d * n, d * m)``
    layer (a $1/d$ parameter ratio).

    Parameters
    ----------
    in_features : int
        Number of hypercomplex input units $n$ per basis component. The
        incoming real feature width must equal ``d * in_features``.
    out_features : int
        Number of hypercomplex output units $m$ per basis component. The
        emitted real feature width equals ``d * out_features``.
    algebra : str or Algebra
        Algebra instance or name/alias accepted by ``get_algebra``.
    bias : bool, default=True
        If ``True``, adds a learnable bias of shape ``(d, out_features)``
        initialized to zeros.

    Attributes
    ----------
    algebra : Algebra
        Resolved algebra specification.
    component_count : int
        Number of real components $d$ in ``algebra``.
    weight : torch.nn.Parameter
        Component weight kernels of shape ``(d, in_features, out_features)``.
    bias : torch.nn.Parameter or None
        Component bias tensor of shape ``(d, out_features)``, or ``None``.

    Notes
    -----
    **Shape:**
        - Input: ``(*, d * in_features)`` where ``*`` is any number of leading
          dimensions (for example ``(batch, steps, d * in_features)``).
        - Output: ``(*, d * out_features)`` with leading axes preserved.

    Examples
    --------
    >>> import torch
    >>> from hypercast4d.layers import HyperDense
    >>> layer = HyperDense(in_features=2, out_features=3, algebra="quaternion")
    >>> x = torch.randn(4, 10, 8)  # 4 components * 2 input units = 8
    >>> out = layer(x)
    >>> tuple(out.shape)  # 4 components * 3 output units = 12
    (4, 10, 12)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        algebra: str | Algebra,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.algebra = get_algebra(algebra) if isinstance(algebra, str) else algebra
        self.component_count = self.algebra.component_count
        self.weight = nn.Parameter(
            torch.empty(self.component_count, in_features, out_features)
        )
        self.bias = (
            nn.Parameter(torch.empty(self.component_count, out_features))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize each component kernel with Glorot normal and zero bias.

        Matches the supplementary TensorFlow/Keras ``HyperDense`` reference
        implementation in Kycia & Niemczynowicz (2025), applying independent
        Xavier-normal initialization to each ``(in_features, out_features)``
        slice ``weight[a]``.
        """
        for component in self.weight:
            nn.init.xavier_normal_(component)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the hypercomplex linear projection along the last dimension.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(*, component_count * in_features)``.

        Returns
        -------
        torch.Tensor
            Projected tensor of shape ``(*, component_count * out_features)``.

        Raises
        ------
        ValueError
            If ``inputs.shape[-1] != component_count * in_features``.
        """
        expected = self.component_count * self.in_features
        if inputs.shape[-1] != expected:
            raise ValueError(
                f"Expected final dimension {expected}, got {inputs.shape[-1]}"
            )
        components = inputs.reshape(
            *inputs.shape[:-1], self.component_count, self.in_features
        )
        constants = self.algebra.constants.to(inputs.device, inputs.dtype)
        # ``components`` supplies the left factor (index b) and ``weight`` the
        # right factor (index a), matching the x * W block-matrix orientation
        # in the paper's archived HyperDense layer.
        outputs = torch.einsum(
            "bac,aio,...bi->...co", constants, self.weight, components
        )
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs.reshape(
            *inputs.shape[:-1], self.component_count * self.out_features
        )

    def extra_repr(self) -> str:
        """Return module summary string for ``repr(self)``."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"algebra={self.algebra.name}, bias={self.bias is not None}"
        )


class ShapePreservingHyperDense(nn.Module):
    """Auto-fitting wrapper that preserves arbitrary last-axis feature widths.

    Given an incoming width ``C`` and an algebra of dimension ``d``, this
    wrapper computes ``units = ceil(C / d)`` and ``padded_width = units * d``.
    During ``forward``, it right-pads the last axis with ``padded_width - C``
    zeros, applies ``HyperDense(units, units, algebra, bias)``, and slices the
    first ``C`` features back out as a contiguous tensor.

    Parameters
    ----------
    width : int
        Exact incoming and outgoing feature width ``C`` on the last tensor axis.
    algebra : str or Algebra
        Algebra instance or name/alias accepted by
        [`get_algebra`][hypercast4d.algebras.get_algebra].
    bias : bool, default=True
        Whether the underlying ``HyperDense`` layer includes a bias term.

    Attributes
    ----------
    width : int
        Unpadded input/output feature width ``C``.
    padded_width : int
        Internal multiple-of-``d`` width ``ceil(C / d) * d``.
    hyper : HyperDense
        Underlying square hypercomplex layer of size ``(units, units)``.

    Examples
    --------
    >>> import torch
    >>> from hypercast4d.layers import ShapePreservingHyperDense
    >>> block = ShapePreservingHyperDense(width=7, algebra="quaternion")
    >>> block.shape_fit["padded_width"], block.shape_fit["units"]
    (8, 2)
    >>> tuple(block(torch.randn(2, 16, 7)).shape)
    (2, 16, 7)
    """

    def __init__(
        self, width: int, algebra: str | Algebra, bias: bool = True
    ) -> None:
        super().__init__()
        self.width = width
        resolved = get_algebra(algebra) if isinstance(algebra, str) else algebra
        units = (width + resolved.component_count - 1) // resolved.component_count
        self.hyper = HyperDense(units, units, resolved, bias)
        self.padded_width = units * resolved.component_count

    @property
    def shape_fit(self) -> dict[str, int]:
        """Return diagnostic metadata describing the pad-and-crop adaptation."""
        return dict(
            axis=-1,
            input_width=self.width,
            padded_width=self.padded_width,
            units=self.hyper.in_features,
            output_width=self.width,
            padding=self.padded_width - self.width,
            crop=self.padded_width - self.width,
        )

    @staticmethod
    def validate_input(inputs: Any) -> None:
        """Verify that ``inputs`` is a real floating-point tensor of rank >= 2.

        Parameters
        ----------
        inputs : Any
            Candidate input object at the graph connection.

        Raises
        ------
        ValueError
            If ``inputs`` is not a real floating-point ``torch.Tensor`` with at
            least batch and feature dimensions and a non-empty last axis.
        """
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim < 2
            or not inputs.is_floating_point()
        ):
            raise ValueError(
                "HyperDense auto-fit requires a real floating-point tensor with batch and feature axes; "
                "tuples, scalars, indices and complex FFT tensors are unsupported at this connection"
            )
        if inputs.shape[-1] < 1:
            raise ValueError("HyperDense auto-fit requires a nonempty last axis")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Zero-pad ``inputs``, apply ``self.hyper``, and crop back to ``width``.

        Parameters
        ----------
        inputs : torch.Tensor
            Real floating-point tensor of shape ``(*, width)`` with rank >= 2.

        Returns
        -------
        torch.Tensor
            Contiguous tensor of identical shape ``(*, width)``.

        Raises
        ------
        ValueError
            If ``inputs`` fails :meth:`validate_input` or ``inputs.shape[-1]``
            differs from ``self.width``.
        """
        self.validate_input(inputs)
        if inputs.shape[-1] != self.width:
            raise ValueError(
                f"HyperDense auto-fit was constructed for last-axis width {self.width}, "
                f"got {inputs.shape[-1]}; dynamic width changes require a different connection"
            )
        padded = torch.nn.functional.pad(inputs, (0, self.padded_width - self.width))
        # Cropping leaves gaps in the padded strides. Downstream TSLib view()
        # operations must be able to flatten/reshape the preserved dimensions.
        return self.hyper(padded)[..., : self.width].contiguous()
