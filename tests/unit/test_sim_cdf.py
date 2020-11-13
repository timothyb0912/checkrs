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

from checkrs import ChartData
from checkrs import View
from checkrs import ViewSimCDF

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
            current_data["id_sim"] = i
            dataframes.append(current_data)
        data = pd.concat(dataframes, ignore_index=True)

        metadata = {
            "target": "target",
            "observed": "observed",
            "id_sim": "id_sim",
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
        "id_sim".
        """
        chart_dict = self.chart_altair.to_dict()
        num_simulations = self.data.data["id_sim"].unique().size
        self.assertEqual(len(chart_dict["layer"]), num_simulations)

    def test_num_layers_plotnine(self):
        """
        We should have one layer per CDF and one CDF per unique value of
        "id_sim".
        """
        num_simulations = self.data.data["id_sim"].unique().size
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
        id_col_sim = self.data.metadata["id_sim"]
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
        # Is the x and y-axis label correct?
        chart = self.chart.from_chart_data(self.data)
        chart_plotnine = self.chart_p9
        self.assertEqual(chart_plotnine.labels["x"], chart.theme.label_x)
        self.assertEqual(chart_plotnine.labels["y"], chart.theme.label_y)

    def test_chart_uses_plot_theme_title_altair(self):
        chart = self.chart.from_chart_data(self.data)
        chart.theme.title = "A Fancy Title!"
        chart_dict = chart.draw(backend="altair").to_dict()
        self.assertEqual(chart_dict["title"], chart.theme.title,)

    def test_chart_uses_plot_theme_title_plotnine(self):
        chart = self.chart.from_chart_data(self.data)
        chart.theme.title = "A Test!"
        chart_plotnine = chart.draw(backend="plotnine")
        self.assertEqual(chart_plotnine.labels["title"], chart.theme.title)

    def test_chart_uses_plot_theme_rotations_altair(self):
        chart = self.chart.from_chart_data(self.data)
        chart_dict = self.chart_altair.to_dict()
        self.assertEqual(
            chart_dict["config"]["axisY"]["titleAngle"],
            chart.theme.rotation_y,
        )

    def test_chart_uses_plot_theme_rotations_plotnine(self):
        chart = self.chart.from_chart_data(self.data)
        chart_plotnine = self.chart_p9
        self.assertEqual(
            chart_plotnine.theme
                .themeables["axis_title_y"]
                .properties["rotation"],
            chart.theme.rotation_y,
        )

    def test_chart_uses_plot_theme_padding_altair(self):
        chart = self.chart.from_chart_data(self.data)
        chart_dict = self.chart_altair.to_dict()
        self.assertEqual(
            chart_dict["config"]["axisY"]["titlePadding"],
            chart.theme.padding_y_altair,
        )

    def test_chart_uses_plot_theme_padding_plotnine(self):
        chart = self.chart.from_chart_data(self.data)
        chart_plotnine = self.chart_p9
        self.assertEqual(
            chart_plotnine.theme
                .themeables["axis_title_y"]
                .properties["margin"]["r"],
            chart.theme.padding_y_plotnine,
        )

    def test_chart_uses_plot_theme_figsize_altair(self):
        chart = self.chart.from_chart_data(self.data)
        chart_dict = self.chart_altair.to_dict()
        self.assertEqual(chart_dict["width"], chart.theme.width_pixels)
        self.assertEqual(chart_dict["height"], chart.theme.height_pixels)

    def test_chart_uses_plot_theme_figsize_plotnine(self):
        chart = self.chart.from_chart_data(self.data)
        chart_plotnine = self.chart_p9
        self.assertEqual(
            chart_plotnine.theme.themeables["figure_size"].properties["value"],
            (chart.theme.width_inches, chart.theme.height_inches),
        )

    def test_chart_uses_plot_theme_colors_altair(self):
        chart = self.chart.from_chart_data(self.data)
        specified_colors = set(
            (chart.theme.color_observed, chart.theme.color_simulated,)
        )

        chart_dict = self.chart_altair.to_dict()
        plotted_colors = set()
        for layer in chart_dict["layer"]:
            plotted_colors.update(layer["encoding"]["color"]["scale"]["range"])

        self.assertEqual(len(plotted_colors), 2)
        self.assertEqual(plotted_colors, specified_colors)

    def test_chart_uses_plot_theme_colors_plotnine(self):
        chart = self.chart.from_chart_data(self.data)
        chart_plotnine = self.chart_p9
        num_kinds = self.data.data["observed"].unique().size
        specified_color_set = set(
            (chart.theme.color_observed, chart.theme.color_simulated)
        )
        observed_scales = tuple(
            set(scale.palette(num_kinds)) for scale in chart_plotnine.scales
        )
        self.assertTrue(specified_color_set in observed_scales)

    def test_chart_uses_plot_theme_fontsize_altair(self):
        chart = self.chart.from_chart_data(self.data)
        chart.theme.title = "A Fancy Title!"
        chart_dict = chart.draw(backend="altair").to_dict()
        # Collect all fontsizes
        fontsizes = set()
        fontsizes.add(chart_dict["config"]["axisX"]["labelFontSize"])
        fontsizes.add(chart_dict["config"]["axisX"]["titleFontSize"])
        fontsizes.add(chart_dict["config"]["axisY"]["labelFontSize"])
        fontsizes.add(chart_dict["config"]["axisY"]["titleFontSize"])
        if "title" in chart_dict:
            fontsizes.add(chart_dict["config"]["title"]["fontSize"])
        # Make sure they equal the specified fontsize
        fontsizes = list(fontsizes)
        self.assertEqual(len(fontsizes), 1)
        self.assertEqual(fontsizes[0], chart.theme.fontsize)

    def test_chart_uses_plot_theme_fontsize_plotnine(self):
        chart = self.chart.from_chart_data(self.data)
        chart_plotnine = self.chart_p9
        self.assertEqual(
            chart_plotnine.theme
                .themeables["axis_text"]
                .properties["size"],
            chart.theme.fontsize,
        )

    def test_setting_chart_plotting_column(self):
        """
        Ensure we can appropriately change the chart's plotting column.
        """
        data = self.data
        new_plotting_column = "target_copy"

        data.data[new_plotting_column] = data.data[
            self.data.metadata["target"]
        ].copy()
        chart = self.chart.from_chart_data(data)
        chart.set_plotting_col(new_plotting_column)

        chart_dict = chart.draw(backend="altair").to_dict()
        chart_p9 = chart.draw(backend="plotnine")

        for layer in chart_p9.layers:
            self.assertEqual(layer.mapping["x"], new_plotting_column)
        for layer in chart_dict["layer"]:
            self.assertEqual(
                layer["encoding"]["x"]["field"], new_plotting_column
            )

    def test_setting_chart_plotting_column_validation(self):
        """
        Make sure we can't store invalid plotting columns.
        """
        bad_plotting_columns_type = [1.7, None, False]
        bad_plotting_columns_value = ["foo"]
        chart = self.chart.from_chart_data(self.data)
        for column in bad_plotting_columns_type:
            self.assertRaises(
                TypeError,
                chart.set_plotting_col,
                column=column
            )
        for column in bad_plotting_columns_value:
            self.assertRaises(
                ValueError,
                chart.set_plotting_col,
                column=column
            )
