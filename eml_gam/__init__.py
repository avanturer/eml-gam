"""EML-GA²M: gradient-trained symbolic GAM with pairwise interactions."""

from .eml_tree import EMLTree
from .gam import EMLGAM
from .interaction_select import select_pairs
from .primitives import (
    PrimitiveConfig,
    bivariate_atlas_depth2,
    default_atlas,
    univariate_atlas_depth1,
    univariate_atlas_depth2,
    warm_start_tree,
)
from .symbolic import complexity, format_formula, to_numpy_fn, verify_formula
from .train import TrainConfig, train_tree, train_with_multistart
from .utils import CLAMP_VAL, DTYPE, EPS, safe_eml

__all__ = [
    "EMLTree",
    "EMLGAM",
    "TrainConfig",
    "train_tree",
    "train_with_multistart",
    "select_pairs",
    "PrimitiveConfig",
    "default_atlas",
    "univariate_atlas_depth1",
    "univariate_atlas_depth2",
    "bivariate_atlas_depth2",
    "warm_start_tree",
    "format_formula",
    "complexity",
    "to_numpy_fn",
    "verify_formula",
    "safe_eml",
    "DTYPE",
    "EPS",
    "CLAMP_VAL",
]
__version__ = "0.1.0"
