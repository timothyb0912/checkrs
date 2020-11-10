"""
This file tests the Chart protocol's contract.
"""
import unittest
from itertools import product
from typing import List

from checkrs.base import ChartData, View, ViewObject
from checkrs.sim_cdf import ViewSimCDF



class ProtocolTests(unittest.TestCase):
    """
    Testing basic properties of a Checkrs.Chart instance.
    Are the signatures correct?
    """
    temp_dir = "_tmp/"
    charts_all: List[View] = [ViewSimCDF]
    backends: List[str] = ["altair", "plotnine"]

    @property
    def data(self) -> ChartData:
        raise NotImplementedError()

    def test_draw_signature(self):
        """
        GIVEN a chart instantiated with valid_chart_data
        WHEN we call the draw method with any valid keyword-argument
        THEN we receive the appropriate matplotlib.Figure or an altair.Chart
        """
        for backend, view in product(self.backends, self.charts_all):
            chart = view(self.data)
            manipulable_object = chart.draw(backend=backend)
            self.assertIsInstance(manipulable_object, ViewObject)

    def test_save_functionality(self):
        """
        GIVEN a chart instantiated with valid_chart_data
        WHEN we call the save method with any valid keyword-argument
        THEN the appropriate file will be saved to its appropriate location
        """
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)  # Make a directory to hold the test plots
        filename = os.path.join(self.temp_dir, "test_filename")
        extensions = [".png", ".pdf", ".json", ".html"]
        try:
            for ext, chart_class in product(extensions, CHARTS_ALL):
                chart = chart_class(self.data)
                full_path_current = filename + ext
                # Ensure missing file, create the file, ensure existing file
                self.assertFalse(os.path.exists(full_path_current))
                chart.save(full_path_current)
                self.assertTrue(os.path.exists(full_path_current))
        except:
            os.rmdir(temp_dir)  # Clear up test plots even if failure happens
            raise  # Re-raise last exception
