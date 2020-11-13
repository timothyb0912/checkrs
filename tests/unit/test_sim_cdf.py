"""
Test that the SimCDF plot is performing as expected.
"""
import unittest
from itertools import product
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import plotnine as p9

from checkrs.base import ChartData, View
from checkrs.sim_cdf import ViewSimCDF

# The test data is "large" so disable warnings
alt.data_transformers.disable_max_rows()


class ChartAttributeTests(unittest.TestCase):
    """
    Testing expected attributes of a SimCDF plot in altair and plotnine.
    """
    chart: View = ViewSimCDF
    backends: List[str] = [
        "altair",
        "plotnine",
    ]
    extensions: List[str] = [
        ".png",
        ".pdf",
        ".json",
        ".html",
    ]

    @property
    def data(self) -> ChartData:
        np.random.seed(324)
        y_all = np.random.rand(100, 100)
        dataframes = []
        for i in range(1, y_all.shape[1]+1):
            current_data = pd.DataFrame({"target": y_all[:, i-1]})
            current_data["observed"] = True if i == 1 else False
            current_data["id_col_sim"] = i
            dataframes.append(current_data)
        data = pd.concat(dataframes, ignore_index=True)

        metadata = {
            "target": "target",
            "observed": "observed",
            "id_col_sim": "id_col_sim",
        }
        return ChartData(data=data, url=None, metadata=metadata)

    @property
    def chart_p9(self) -> p9.ggplot:
        return self.chart.from_chart_data(self.data).draw(backend="plotnine")

    @property
    def chart_altair(self) -> alt.TopLevelMixin:
        return self.chart.from_chart_data(self.data).draw(backend="altair")

    def test_is_layer_chart_altair(self):
        """
        Altair SimCDF plots should be layer charts! One layer per CDF.
        """
        chart_dict = self.chart_altair.to_dict()
        self.assertTrue("layer" in chart_dict)

    def test_num_layers_altair(self):
        """
        We should have one layer per CDF and one CDF per unique value of
        "id_col_sim".
        """
        chart_dict = self.chart_altair.to_dict()
        num_simulations = self.data.data["id_col_sim"].unique().size
        self.assertEqual(len(chart_dict["layer"]), num_simulations)

    def test_num_layers_plotnine(self):
        """
        We should have one layer per CDF and one CDF per unique value of
        "id_col_sim".
        """
        num_simulations = self.data.data["id_col_sim"].unique().size
        self.assertEqual(len(self.chart_p9.layers), num_simulations)

    def test_layers_are_cdfs_plotnine(self):
        """
        Each layer of the visualization should be displaying a CDF.
        """
        for layer in self.chart_p9.layers:
            self.assertIsInstance(layer.stat, p9.stat_ecdf)

    def test_layers_are_cdfs_altair(self):
        """
        Each layer of the visualization should display a CDF.
        """
        chart_dict = self.chart_altair.to_dict()
        for layer in chart_dict["layer"]:
            # The layer should be a line
            self.assertEqual(layer["mark"], "line")
            # The line should be formed by a cumulative density transform
            density_transforms = tuple(
                transform for transform in layer["transform"]
                if "density" in transform
            )
            self.assertTrue(len(density_transforms) == 1)
            self.assertTrue(density_transforms[0]["cumulative"])

    def test_layers_use_unique_data_altair(self):
        """
        Each layer of the visualization should use a unique slice of the data
        for plotting CDFs.
        """
        filterings = set()
        chart_dict = self.chart_altair.to_dict()
        for layer in chart_dict["layer"]:
            # Make sure each layer a unique filtering of the data
            transforms = layer.get("transform", None)
            current_filtering = tuple(
                transform for transform in transforms
                if "filter" in transform
            )
            current_filtering = (
                current_filtering[0]["filter"] if len(current_filtering) > 0
                else None
            )
            self.assertTrue(current_filtering not in filterings)
            filterings.add(current_filtering)

    def test_layers_use_unique_data_plotnine(self):
        """
        Each layer of the visualization should use a unique slice of the data
        for plotting CDFs.
        """
        sim_ids = set()
        chart = self.chart_p9
        id_col_sim = self.data.metadata["id_col_sim"]
        for layer in chart.layers:
            current_data = layer.data if layer.data is not None else chart.data
            sim_ids_potential = current_data[id_col_sim].unique()
            self.assertEqual(sim_ids_potential.size, 1)
            sim_id_current = sim_ids_potential[0]
            self.assertTrue(sim_id_current not in sim_ids)
            sim_ids.add(sim_id_current)

    def test_layers_use_correct_axes_encodings_altair(self):
        """
        Make sure that we're correctly mapping each layer's data to plot axes
        and colors.
        """
        outcome_col = self.data.metadata["target"]
        observed_col = self.data.metadata["observed"]
        encodings = {
            "x": {"field": outcome_col},
            "y": {"field": "density"},
            "color": {"field": observed_col}
        }
        chart_dict = self.chart_altair.to_dict()
        for layer in chart_dict["layer"]:
            layer_encoding = layer["encoding"]
            for key in encodings:
                self.assertEqual(
                    layer_encoding[key]["field"], encodings[key]["field"]
                )

    def test_layers_use_correct_axes_encodings_plotnine(self):
        """
        Make sure that we're correctly mapping each layer's data to plot axes
        and colors.
        """
        outcome_col = self.data.metadata["target"]
        observed_col = self.data.metadata["observed"]
        for layer in self.chart_p9.layers:
            self.assertEqual(layer.mapping["x"], outcome_col)
            self.assertEqual(layer.mapping["color"], observed_col)
            # Note no need to check y as we know its CDF from stat_ecdf

    def test_chart_uses_plot_theme_labels_altair(self):
        chart = self.chart.from_chart_data(self.data)
        encodings = {
            "x": {
                    "field": chart.theme.plotting_col,
                    "title": chart.theme.label_x,
                },
            "y": {
                    "field": "density",
                    "title": chart.theme.label_y,
                },
        }
        chart_dict = self.chart_altair.to_dict()
        for layer in chart_dict["layer"]:
            layer_encoding = layer["encoding"]
            for key in encodings:
                self.assertEqual(
                    layer_encoding[key]["title"], encodings[key]["title"]
                )

    def test_chart_uses_plot_theme_labels_plotnine(self):
        pass

    def test_chart_uses_plot_theme_title_altair(self):
        pass

    def test_chart_uses_plot_theme_title_altair(self):
        pass

    def test_chart_uses_plot_theme_rotations_altair(self):
        pass

    def test_chart_uses_plot_theme_rotations_plotnine(self):
        pass

    def test_chart_uses_plot_theme_padding_altair(self):
        pass

    def test_chart_uses_plot_theme_padding_plotnine(self):
        pass

    def test_chart_uses_plot_theme_figsize_altair(self):
        pass

    def test_chart_uses_plot_theme_figsize_plotnine(self):
        pass

    def test_chart_uses_plot_theme_colors_altair(self):
        pass

    def test_chart_uses_plot_theme_colors_plotnine(self):
        pass

    def test_chart_uses_plot_theme_fontsize_altair(self):
        pass

    def test_chart_uses_plot_theme_fontsize_plotnine(self):
        pass
