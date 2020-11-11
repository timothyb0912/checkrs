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
            current_data = pd.DataFrame({"y": y_all[:, i-1]})
            current_data["observed"] = True if i == 1 else False
            current_data["id_col_sim"] = i
            dataframes.append(current_data)
        data = pd.concat(dataframes, ignore_index=True)

        metadata = {
            "y": "y", "observed": "observed", "id_col_sim": "id_col_sim"
        }
        return ChartData(data=data, url=None, metadata=metadata)

    def test_is_layer_chart_altair(self):
        """
        Altair SimCDF plots should be layer charts! One layer per CDF.
        """
        chart = self.chart.from_chart_data(self.data).draw(backend="altair")
        chart_dict = chart.to_dict()
        self.assertTrue("layer" in chart_dict)

    def test_num_layers_altair(self):
        """
        We should have one layer per CDF and one CDF per unique value of
        "id_col_sim".
        """
        chart = self.chart.from_chart_data(self.data).draw(backend="altair")
        chart_dict = chart.to_dict()
        num_simulations = self.data.data["id_col_sim"].unique().size
        self.assertEqual(len(chart_dict["layer"]), num_simulations)

    def test_num_layers_plotnine(self):
        """
        We should have one layer per CDF and one CDF per unique value of
        "id_col_sim".
        """
        chart = self.chart.from_chart_data(self.data).draw(backend="plotnine")
        num_simulations = self.data.data["id_col_sim"].unique().size
        self.assertEqual(len(chart.layers), num_simulations)

    def test_layers_are_cdfs_plotnine(self):
        """
        Each layer of the visualization should be displaying a CDF.
        """
        chart = self.chart.from_chart_data(self.data).draw(backend="plotnine")
        for layer in chart.layers:
            self.assertIsInstance(layer.stat, p9.stat_ecdf)

    def test_layers_are_cdfs_altair(self):
        """
        Each layer of the visualization should be displaying a CDF.
        """
        chart = self.chart.from_chart_data(self.data).draw(backend="altair")
        chart_dict = chart.to_dict()
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
