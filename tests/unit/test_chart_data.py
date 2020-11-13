"""
Test that the ChartData object is validated as expected.
"""
import unittest
from itertools import product
from typing import Any, List

import altair as alt
import numpy as np
import pandas as pd
import plotnine as p9

from checkrs import ChartData


class ChartDataTests(unittest.TestCase):
    """
    Testing data validation and instantiation of ChartData.
    """
    num_obs : int = 100

    @property
    def good_data(self) -> ChartData:
        np.random.seed(324)
        y_all = np.random.rand(self.num_obs, 10)
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

    def bad_data(self, kind: str = "metadata") -> List[Any]:
        """
        Get examples of various kinds of invalid data.
        """
        gremlins = []
        if kind == "metadata":
            return self._bad_data_metadata()
        elif kind == "url":
            return self._bad_data_url()
        elif kind == "data":
            return self._bad_data_data()
        return gremlins

    def _bad_data_metadata(self) -> List[Any]:
        """
        Examples of invalid metadata.
        """
        gremlins = [
            None,
            dict(),
            {"target": "target", "observed": "observed"},
            {"target": "target",
             "simulated": "observed",
             "id_sim": "id_sim",
            },
            {"target": "target", "observed": "observed", "id_sim": 1},
            {"target": "target", "observed": "observed", "id_sim": "foo"},
        ]
        return gremlins

    def _bad_data_url(self) -> List[Any]:
        """
        Examples of invalid urls .
        """
        gremlins = [
            "",
            "data.csv",
            "data.jso",
            100
        ]
        return gremlins

    def _bad_data_data(self) -> List[Any]:
        gremlins = [
            10,
            dict(),
            np.ones((5, 4), dtype=np.float32),
            [range(1, 4), range(4, 7), range(7, 10)],
        ]

        df_wrong_col = self.good_data.data.copy()
        df_wrong_col.rename(columns={"target": "y"}, inplace=True)
        gremlins.append(df_wrong_col)

        df_missing_col = self.good_data.data.copy()
        del df_missing_col["observed"]
        gremlins.append(df_missing_col)

        df_unequal_shapes = self.good_data.data.copy()
        df_unequal_shapes.drop(df_unequal_shapes.index[0], inplace=True)
        gremlins.append(df_unequal_shapes)
        return gremlins

    def test_metadata_errors(self):
        """
        Raise a ValueError if we pass bad metadata.
        """
        good_data = self.good_data
        for metadata in self.bad_data(kind="metadata"):
            self.assertRaises(
                ValueError,
                ChartData,
                data=good_data.data,
                url=good_data.url,
                metadata=metadata
            )

    def test_url_errors(self):
        """
        Raise a ValueError if we pass a bad URL.
        """
        good_data = self.good_data
        for url in self.bad_data(kind="url"):
            self.assertRaises(
                ValueError,
                ChartData,
                data=good_data.data,
                url=url,
                metadata=good_data.metadata
            )

    def test_data_errors(self):
        """
        Raise a ValueError if we pass a bad data.
        """
        good_data = self.good_data
        for data in self.bad_data(kind="data"):
            self.assertRaises(
                ValueError,
                ChartData,
                data=data,
                url=good_data.url,
                metadata=good_data.metadata
            )

    def test_from_raw(self):
        """
        Create a ChartData instance from raw data. Should throw no errors.
        """
        np.random.seed(324)
        num_simulations = 10
        y_all = np.random.rand(self.num_obs, num_simulations + 1)
        y = y_all[:, 0]
        y_sim = y_all[:, 1:]
        x = pd.DataFrame(
            {"x":
                np.random.choice([0, 1], p=[0.5, 0.5], size=self.num_obs),
             "x2":
                np.random.choice([3, 4], p=[0.5, 0.5], size=self.num_obs),
            }
        )

        custom_urls = [
            "./temp/fancy_custom_name.json",
            "data.json",
            None,
        ]
        for url in custom_urls:
            data = ChartData.from_raw(
                design=x, targets=y, targets_simulated=y_sim, url=url
            )
            self.assertEqual(data.url, url)
            self.assertEqual(
                data.data[data.metadata["id_sim"]].unique().size,
                num_simulations + 1
            )
            self.assertTrue(all(col in data.data.columns for col in x.columns))
