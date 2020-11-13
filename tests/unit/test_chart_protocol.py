"""
This file tests the Chart protocol's contract.
"""
import os
import shutil
import unittest
from itertools import product
from typing import List

import numpy as np
import pandas as pd

from checkrs import ChartData
from checkrs import View
from checkrs import ViewSimCDF
from checkrs.base import ggplot
from checkrs.base import TopLevelMixin


class ProtocolTests(unittest.TestCase):
    """
    Testing basic properties of a Checkrs.Chart instance.
    Are the signatures correct?
    """
    temp_dir = "_tmp/"
    charts_all: List[View] = [ViewSimCDF]
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
        y_all = np.random.rand(100, 10)
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

    def test_draw_signature(self):
        """
        GIVEN a chart instantiated with valid_chart_data
        WHEN we call the draw method with any valid keyword-argument
        THEN we receive the appropriate matplotlib.Figure or an altair.Chart
        """
        for backend, view in product(self.backends, self.charts_all):
            chart = view.from_chart_data(data=self.data)
            manipulable_object = chart.draw(backend=backend)
            self.assertIsInstance(manipulable_object, (ggplot, TopLevelMixin))

    def test_save_functionality(self):
        """
        GIVEN a chart instantiated with valid_chart_data
        WHEN we call the save method with any valid keyword-argument
        THEN the appropriate file will be saved to its appropriate location
        AND we will be returned a boolean indicating saving success
        """
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)  # Make a directory to hold the test plots
        filename = os.path.join(self.temp_dir, "test_filename")
        try:
            for ext, view in product(self.extensions, self.charts_all):
                chart = view.from_chart_data(data=self.data)
                full_path_current = filename + ext
                # Ensure missing file, create the file, ensure existing file
                self.assertFalse(os.path.exists(full_path_current))
                result = chart.save(full_path_current)
                self.assertIsInstance(result, bool)
                self.assertTrue(os.path.exists(full_path_current))
        finally:
            # Clear up test plots even if failure happens
            shutil.rmtree(self.temp_dir, ignore_errors=True)
