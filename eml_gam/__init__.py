"""EML-GA²M: gradient-trained symbolic GAM with pairwise interactions."""

from .atlas_expansion import AtlasCandidate, aees_recover, aees_search
from .eml_tree import EMLTree
from .gam import EMLGAM
from .interaction_select import select_pairs
from .primitives import (
    PrimitiveConfig,
    bivariate_atlas_depth2,
    default_atlas,
    rank_atlas_candidates,
    univariate_atlas_depth1,
    univariate_atlas_depth2,
    warm_start_tree,
)
from .sheffer import PsiTree, safe_psi
from .symbolic import complexity, format_formula, to_numpy_fn, verify_formula
from .train import TrainConfig, train_tree, train_with_multistart
from .utils import CLAMP_VAL, DTYPE, EPS, safe_eml

__all__ = [
    "EMLTree",
    "EMLGAM",
    "PsiTree",
    "safe_psi",
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
    "rank_atlas_candidates",
    "AtlasCandidate",
    "aees_search",
    "aees_recover",
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
