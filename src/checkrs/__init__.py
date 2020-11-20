# -*- coding: utf-8 -*-
"""
Tools for simulation-based model checking and diagnostics.
"""
from __future__ import absolute_import

from .base import ChartData  # noqa: F401
from .base import View  # noqa: F401
from .cont_scalars import plot_continous_scalars  # noqa: F401
from .disc_scalars import plot_discrete_scalars  # noqa: F401
from .marginal import plot_smoothed_marginal  # noqa: F401
from .market import plot_simulated_market_shares  # noqa: F401
from .reliability import plot_smoothed_reliability  # noqa: F401
from .sim_cdf import plot_simulated_cdfs  # noqa: F401
from .sim_cdf import ViewSimCDF  # noqa: F401
from .sim_kde import plot_simulated_kdes  # noqa: F401
from .utils import compute_predictive_log_likelihoods  # noqa: F401
from .utils import compute_predictive_mse  # noqa: F401
from .utils import is_categorical  # noqa: F401
from .utils import progress  # noqa: F401
from .utils import simulate_choice_vector  # noqa: F401

__version__ = "0.2.0"
