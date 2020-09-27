# -*- coding: utf-8 -*-
"""
Tools for simulation-based model checking and diagnostics.
"""
from __future__ import absolute_import

from pkg_resources import DistributionNotFound, get_distribution

from .cont_scalars import plot_continous_scalars
from .disc_scalars import plot_discrete_scalars
from .marginal import plot_smoothed_marginal
from .market import plot_simulated_market_shares
from .reliability import plot_smoothed_reliability
from .sim_cdf import plot_simulated_cdfs
from .sim_kde import plot_simulated_kdes
from .utils import (
    compute_predictive_log_likelihoods,
    compute_predictive_mse,
    is_categorical,
    progress,
    simulate_choice_vector,
)

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
