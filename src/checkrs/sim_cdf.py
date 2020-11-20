# -*- coding: utf-8 -*-
"""
Functions for plotting simulated vs observed cumulative distribution functions.
"""
from __future__ import absolute_import

import os
from typing import (
    Dict, Iterable, List, Optional
)

import altair as alt
import attr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sbn

import checkrs.base as base
from checkrs.plot_utils import (
    _choice_evaluator,
    _label_despine_save_and_show_plot,
    _plot_single_cdf_on_axis,
    _thin_rows,
)
from checkrs.utils import progress

try:
    # in Python 3 range returns an iterator instead of list
    # to maintain backwards compatibility use "old" version of range
    from past.builtins import range
except ImportError:
    pass

# Set the plotting style
sbn.set_style("darkgrid")


def plot_simulated_cdfs(
    sim_y,
    orig_df,
    filter_idx,
    col_to_plot,
    choice_col,
    sim_color="#a6bddb",
    orig_color="#045a8d",
    choice_condition=1,
    thin_pct=None,
    fig_and_ax=None,
    label="Simulated",
    title=None,
    bar_alpha=0.5,
    bar_color="#fee391",
    n_traces=None,
    rseed=None,
    show=True,
    figsize=(10, 6),
    fontsize=12,
    xlim=None,
    ylim=None,
    output_file=None,
    dpi=500,
    **kwargs,
):
    """
    Plots an observed cumulative density function (CDF) versus the simulated
    versions of that same CDF.

    Parameters
    ----------
    sim_y : 2D ndarray.
        The simulated outcomes. All elements should be zeros or ones. There
        should be one column for every set of simulated outcomes. There should
        be one row for every row of one's dataset.
    orig_df : pandas DataFrame.
        The dataframe containing the data used to estimate one's model. Should
        have the same number of rows as `sim_y`.
    filter_idx : 1D ndarray of booleans.
        Should have the same number of rows as `orig_df`. Will denote the rows
        that should be used to compute the CDF if their outcome is
        `choice_condition`.
    col_to_plot : str.
        A column in `orig_df` whose data will be used to compute the KDEs.
    choice_col : str.
        The column in `orig_df` containing the data on the original outcomes.
    sim_color, orig_color : valid 'color' argument for matplotlib, optional.
        The colors that will be used to plot the simulated and observed CDFs,
        respectively. Default is `sim_color == '#a6bddb'` and
        `orig_color == '#045a8d'`.
    choice_condition : `{0, 1}`, optional.
        Denotes the outcome class that we wish to plot the CDFs for. If
        `choice_condition == 1`, then we will plot the CDFs for those where
        `sim_y == 1` and `filter_idx == True`. If `choice_condition == 0`, we
        will plot the CDFs for those rows where `sim_y == 0` and
        `filter_idx == True`. Default == 1.
    fig_and_ax : list of matplotlib figure and axis, or `None`, optional.
        Determines whether a new figure will be created for the plot or whether
        the plot will be drawn on the passed Axes object. If None, a new figure
        will be created. Default is `None`.
    label : str or None, optional.
        The label for the simulated CDFs. If None, no label will be displayed.
        Default = 'Simulated'.
    title : str or None, optional.
        The plot title. If None, no title will be displayed. Default is None.
    bar_alpha : float in (0.0, 1.0), optional.
        Denotes the opacity of the bar used to denote the proportion of
        simulations where no observations had `sim_y == choice_condition`.
        Higher values lower the bar's transparency. `0` leads to an invisible
        bar. Default == 0.5.
    bar_color : valid 'color' argument for matplotlib, optional.
        The color that will be used to plot the bar that shows the proportion
        of simulations where no observations had `sim_y == choice_condition`.
        Default is '#fee391'.
    thin_pct : float in (0.0, 1.0) or None, optional.
        Determines the percentage of the data (rows) to be used for plotting.
        If None, the full dataset will be used. Default is None.
    n_traces : int or None, optional.
        Should be less than `sim_y.shape[1]`. Denotes the number of simulated
        choices to randomly select for plotting. If None, all columns of
        `sim_y` will be used for plotting. Default is None.
    rseed : int or None, optional.
        Denotes the random seed to be used when selecting `n_traces` columns
        for plotting. This is useful for reproducing an exact plot when using
        `n_traces`. If None, no random seed will be set. Default is None.
    show : bool, optional.
        Determines whether `fig.show()` will be called after the plots have
        been drawn. Default is True.
    figsize : 2-tuple of ints, optional.
        If a new figure is created for this plot, this kwarg determines the
        width and height of the figure that is created. Default is `(5, 3)`.
    fontsize : int or None, optional.
        The fontsize to be used in the plot. Default is 12.
    xlim, ylim : 2-tuple of ints or None, optional.
        Denotes the extent that will be set on the x-axis and y-axis,
        respectively, of the matplotlib Axes instance that is drawn on. If
        None, then the extent will not be manually altered. Default is None.
    output_file : str, or None, optional.
        Denotes the relative or absolute filepath (including the file format)
        that is to be used to save the plot. If None, the plot will not be
        saved to file. Default is None.
    dpi : positive int, optional.
        Denotes the number of 'dots per inch' for the saved figure. Will only
        be used if `output_file is not None`. Default == 500.
    kwargs : passed to `ax.plot` call in matplotlib.

    Returns
    -------
    None.
    """
    # Filter the data
    filtered_sim_y = sim_y[filter_idx, :]
    filtered_orig_df = orig_df.loc[filter_idx, :]

    if rseed is not None:
        np.random.seed(rseed)

    if n_traces is not None:
        selected_cols = np.random.choice(
            filtered_sim_y.shape[1], size=n_traces, replace=False
        )
        filtered_sim_y = filtered_sim_y[:, selected_cols]

    if thin_pct is not None:
        # Randomly select rows to be retained for plotting
        selected_rows = _thin_rows(filtered_sim_y, thin_pct)
        # Filter the simulated-y, df, and filtering values
        filtered_sim_y = filtered_sim_y[selected_rows, :]
        filtered_orig_df = filtered_orig_df.iloc[selected_rows, :]

    sample_iterator = progress(
        range(filtered_sim_y.shape[1]), desc="Calculating CDFs"
    )

    # Get the original values
    orig_choices = filtered_orig_df[choice_col].values

    orig_plotting_idx = _choice_evaluator(orig_choices, choice_condition)
    orig_plotting_vals = filtered_orig_df.loc[
        orig_plotting_idx, col_to_plot
    ].values

    if fig_and_ax is None:
        fig, axis = plt.subplots(1, figsize=figsize)
        fig_and_ax = [fig, axis]
    else:
        fig, axis = fig_and_ax

    # Count simulated data with no obs meeting the choice and filter conditions
    num_null_choices = 0

    # store the minimum and maximum x-values
    min_x, max_x = orig_plotting_vals.min(), orig_plotting_vals.max()

    for i in sample_iterator:
        current_choices = filtered_sim_y[:, i]

        # Determine the final rows to use for plotting
        plotting_idx = _choice_evaluator(current_choices, choice_condition)

        if plotting_idx.sum() == 0:
            num_null_choices += 1
            continue

        # Get the values for plotting
        current_plotting_vals = filtered_orig_df.loc[
            plotting_idx, col_to_plot
        ].values

        # Update the plot extents
        min_x = min(current_plotting_vals.min(), min_x)
        max_x = max(current_plotting_vals.max(), max_x)

        _plot_single_cdf_on_axis(
            current_plotting_vals, axis, color=sim_color, alpha=0.5, **kwargs
        )

    # Plot the originally observed relationship
    _plot_single_cdf_on_axis(
        orig_plotting_vals,
        axis,
        color=orig_color,
        label="Observed",
        alpha=1.0,
        **kwargs,
    )

    if num_null_choices > 0:
        num_null_pct = num_null_choices / float(filtered_sim_y.shape[1])
        null_pct_density_equivalent = axis.get_ylim()[1] * num_null_pct
        null_label = "'No Obs' Simulations: {:.2%}".format(num_null_pct)
        axis.bar(
            [0],
            [null_pct_density_equivalent],
            width=0.1 * np.ptp(orig_plotting_vals),
            align="edge",
            alpha=bar_alpha,
            color=bar_color,
            label=null_label,
        )

    if label is not None:
        _patch = mpatches.Patch(color=sim_color, label=label)
        current_handles, current_labels = axis.get_legend_handles_labels()
        current_handles.append(_patch)
        current_labels.append(label)

        axis.legend(
            current_handles, current_labels, loc="best", fontsize=fontsize
        )

    # set the plot extents
    if xlim is None:
        axis.set_xlim((min_x, max_x))
    else:
        axis.set_xlim(xlim)

    if ylim is not None:
        axis.set_ylim(ylim)

    # Take care of boilerplate plotting necessities
    _label_despine_save_and_show_plot(
        x_label=col_to_plot,
        y_label="Cumulative\nDensity\nFunction",
        fig_and_ax=fig_and_ax,
        fontsize=fontsize,
        y_rot=0,
        y_pad=40,
        title=title,
        output_file=output_file,
        show=show,
        dpi=dpi,
    )
    return None


@attr.s
class ViewSimCDF(base.View):
    data: pd.DataFrame = attr.ib()
    _url: str = attr.ib()
    _metadata: Dict[str, str] = attr.ib()
    theme : base.PlotTheme = attr.ib()

    def set_plotting_col(self, column: str) -> bool:
        """
        Raises ValueError if `column` not in `data`.
        """
        if not isinstance(column, str):
            msg = "`column` MUST be a string."
            raise TypeError(column)
        if column not in self.data.columns:
            msg = "`column` not in `data.columns`"
            raise ValueError(msg)
        self.theme.plotting_col = column
        return True

    @classmethod
    def from_chart_data(cls, data: base.ChartData) -> "ViewSimCDF":
        """
        Instantiates the simulated CDF chart from the given `ChartData`.
        """
        return cls(
            data=data.data,
            url=data.url,
            metadata=data.metadata,
            theme=base.PlotTheme(
                label_y="Cumulative\nDistribution\nFunction",
                plotting_col=data.metadata["target"],
            ),
        )

    def draw(self, backend: str) -> base.ViewObject:
        """
        Specifies the view of the data using `backend`.
        """
        if backend == "plotnine":
            return self.draw_plotnine()
        elif backend == "altair":
            return self.draw_altair()
        else:
            raise ValueError("`backend` MUST == 'plotnine'.")

    def _get_sim_ids(self) -> List[int]:
        # Note [::-1] puts id_sim = 1 on top (id_sim = 1 is last).
        # Hopefully its the observed line
        sim_ids = np.sort(
            self.data[self._metadata["id_sim"]].unique()
        ).tolist()[::-1]
        return sim_ids

    def draw_plotnine(self) -> p9.ggplot:
        """
        Specifies the plot using plotnine.
        """
        sim_ids = self._get_sim_ids()

        # Add the data to the plot
        chart = p9.ggplot()
        for idx in progress(sim_ids):
            chart = chart + self.create_single_cdf_line_plotnine(idx)

        # Format the plot
        chart = self.format_view_plotnine(chart)
        return chart

    def create_single_cdf_line_plotnine(self, id_sim: int) -> p9.ggplot:
        """
        Specifies a singe CDF line on the plot using plotnine.
        """
        id_col_sim = self._metadata["id_sim"]
        observed_col = self._metadata["observed"]
        return p9.stat_ecdf(
                mapping=p9.aes(
                    x=self.theme.plotting_col,
                    color=observed_col,
                    alpha=observed_col,
                ),
                data=self.data.loc[self.data[id_col_sim] == id_sim],
            )

    def draw_altair(self) -> alt.TopLevelMixin:
        """
        Specifies the plot through Altair.
        """
        sim_ids = self._get_sim_ids()

        # Add the data to the plot
        chart = self.create_single_cdf_line_altair(sim_ids[0])
        for idx in progress(sim_ids[1:]):
            chart += self.create_single_cdf_line_altair(idx)

        # Format the plot
        chart = self.format_view_altair(chart)
        return chart

    def create_single_cdf_line_altair(self, id_sim: int) -> alt.TopLevelMixin:
        """
        Specifies a singe CDF line on the plot using Altair.
        """
        # Get data and metadata
        current_data = self._url if self._url is not None else self.data
        id_col_sim = self._metadata["id_sim"]
        observed_col = self._metadata["observed"]

        # Declare mappings of data to x-axes, y-axes, color and opacity
        observed_domain = [True, False]
        color_range = [self.theme.color_observed, self.theme.color_simulated]
        opacity_range = [1, 0.5]

        encoding_x = alt.X(
            self.theme.plotting_col,
            type="quantitative",
            title=self.theme.label_x,
        )
        encoding_y = alt.Y(
            "density", type="quantitative", title=self.theme.label_y,
        )
        encoding_color = alt.Color(
            observed_col,
            type="nominal",
            scale=alt.Scale(domain=observed_domain, range=color_range,),
        )
        encoding_opacity = alt.Opacity(
            observed_col,
            type="nominal",
            scale=alt.Scale(domain=observed_domain, range=opacity_range,),
        )

        # Create the single cdf chart by filtering, transforming, and encoding
        # data to the lines on the plot.
        chart = (
            alt.Chart(current_data)
                .transform_filter(alt.datum[id_col_sim] == id_sim)
                .transform_density(
                    self.theme.plotting_col,
                    as_=[self.theme.plotting_col, "density"],
                    groupby=[observed_col],
                    cumulative=True,
                    steps=25,
                )
                .mark_line()
                .encode(
                    encoding_x,
                    encoding_y,
                    encoding_color,
                    encoding_opacity,
                )
        )
        return chart

    def format_view_plotnine(self, chart: p9.ggplot) -> p9.ggplot:
        """
        Apply chart formatting options from `self.theme`.
        """
        figure_size = (self.theme.width_inches, self.theme.height_inches)

        chart = (
            chart
            + p9.theme(
                axis_text=p9.element_text(size=self.theme.fontsize),
                axis_title_y=p9.element_text(
                    rotation=self.theme.rotation_y,
                    margin={"r": self.theme.padding_y_plotnine, "units": "pt"},
                ),
                figure_size=figure_size,
                dpi=self.theme.dpi_print,
            )
            + p9.xlab(self.theme.label_x)
            + p9.ylab(self.theme.label_y)
            + p9.scale_color_manual(
                (self.theme.color_simulated, self.theme.color_observed),
                labels=p9.utils.waiver()
            )
            + p9.scale_alpha_manual(
                (0.5, 1),
                labels=p9.utils.waiver()
            )
        )
        if self.theme.title is not None:
            chart = chart + p9.ggtitle(self.theme.title)
        return chart

    def format_view_altair(
        self, chart: alt.TopLevelMixin
    ) -> alt.TopLevelMixin:
        """
        Apply chart formatting options from `self.theme`.
        """
        chart = (
            chart.configure_axisX(
                labelFontSize=self.theme.fontsize,
                labelAngle=self.theme.rotation_x_ticks,
                titleFontSize=self.theme.fontsize,
            ).configure_axisY(
                labelFontSize=self.theme.fontsize,
                titleFontSize=self.theme.fontsize,
                titleAngle=self.theme.rotation_y,
                titlePadding=self.theme.padding_y_altair,
            ).properties(
                width=self.theme.width_pixels,
                height=self.theme.height_pixels,
            )
        )
        if self.theme.title is not None:
            chart = chart.properties(
                width=self.theme.width_pixels,
                height=self.theme.height_pixels,
                title=self.theme.title,
            ).configure_title(fontSize=self.theme.fontsize)
        return chart

    def save(self, filename: str) -> bool:
        """
        Saves the view of the data using the appropriate backend for the
        filename's extension. Returns True if saving succeeded.
        """
        ext = os.path.splitext(filename)[1]
        if ext not in base.EXTENSIONS:
            raise ValueError(f"Format MUST be in {base.EXTENSIONS}")
        if ext in base.EXTENSIONS_PLOTNINE:
            chart = self.draw_plotnine()
        elif ext in base.EXTENSIONS_ALTAIR:
            chart = self.draw_altair()
        chart.save(filename)
        return True
