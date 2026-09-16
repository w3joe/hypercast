import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from analyze_modal_l4_robustness import block_means, relative_effect


def test_block_bootstrap_preserves_pairing_and_known_relative_effect():
    values = np.arange(1, 81, dtype=float)
    samples = block_means(np.column_stack([values, 2*values, values]), 13,
        np.random.default_rng(123), resamples=207)
    np.testing.assert_allclose(relative_effect(samples[:,0],samples[:,1]),50)
    np.testing.assert_allclose(relative_effect(samples[:,0],samples[:,2]),0)
    assert samples.shape == (207,3) and samples[:,0].std() > 0


def test_long_circular_blocks_reproduce_constant_columns():
    samples = block_means(np.ones((7,2)), 30, np.random.default_rng(4),resamples=19)
    np.testing.assert_allclose(samples,1)
