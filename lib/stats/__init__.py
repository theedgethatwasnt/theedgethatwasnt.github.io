"""Statistical gates for strategy deployment.

See research/MCMC_INTEGRATION_PLAN.md for the full plan. Phase 1 modules:
  - perm_test: permutation p-values (pair-shuffle / time-shuffle / block)
  - bootstrap_sharpe: block-bootstrap Sharpe confidence intervals

Phase 2 (not yet):
  - mcmc_probe: Metropolis-Hastings plateau probe for trained NN weights
"""

from .perm_test import permutation_pvalue
from .bootstrap_sharpe import block_bootstrap_sharpe, sharpe_ci_report

__all__ = [
    "permutation_pvalue",
    "block_bootstrap_sharpe",
    "sharpe_ci_report",
]
