"""
Basic objects that serve as templates for specialized functions and classes.
"""
from typing import (
    Dict, Optional, Union
)
from typing_extensions import Protocol

import attr
import pandas as pd
from plotnine import ggplot
from altair import TopLevelMixin


EXTENSIONS_PLOTNINE = set((".png", ".pdf"))
EXTENSIONS_ALTAIR = set((".html", ".json"))
EXTENSIONS = EXTENSIONS_PLOTNINE | EXTENSIONS_ALTAIR

ViewObject = Union[ggplot, TopLevelMixin]


@attr.s
class ChartData:
    """
    The simulated data we wish to visualize.

    Parameters
    ----------
    data : optional, pandas DataFrame or None
        If not None, should contain the data that we wish to visualize. If
        None, then `url` should be a string with extension `.json`.
    url : optional, str or None.
        If not None,  should be a string with extension `.json`.
    metadata : dict.
        Keys and values should both be strings. Should contain, at minimum, the
        following keys: {"observed", "id_col_sim", "target"}. These should map
        to values that are column headings in `data`. Respectively, these
        columns should denote whether the data was observed or simulated, the
        id of the simulation, and the outcome values.
    """
    data: Optional[pd.DataFrame] = attr.ib()
    url: Optional[str] = attr.ib()
    metadata: Dict[str, str] = attr.ib()

    @data.validator
    def _check_data_type(self, attribute, value) -> bool:
        # Check type
        if isinstance(value, pd.DataFrame) or value is None:
            pass
        else:
            msg = "`data` MUST be a pandas DataFrame or None"
            raise ValueError(msg)
        return True

    @data.validator
    def _check_data_columns(self, attribute, value) -> bool:
        self._check_metadata("metadata", self.metadata)
        # Check columns
        id_col_sim = self.metadata["id_col_sim"]
        needed_cols = [
            id_col_sim, self.metadata["observed"], self.metadata["target"]
        ]
        has_needed_cols = all(
            (column in self.data.columns for column in needed_cols)
        )
        if not has_needed_cols:
            msg = f"`data.columns` MUST contain:\n{needed_cols}"
            raise ValueError(msg)
        return True

    @data.validator
    def _check_data_shapes(self, attribute, value) -> bool:
        self._check_metadata("metadata", self.metadata)
        # Check shape
        shape = {}
        id_col_sim = self.metadata["id_col_sim"]
        for idx in self.data[id_col_sim].unique():
            current_shape = self.data.loc[
                self.data[id_col_sim] == idx
            ].shape[0]
            orig_shape = shape.get("value", current_shape)
            shape["value"] = current_shape
            if current_shape != orig_shape:
                msg = "Unequal shapes across simulation id's."
                raise ValueError(msg)
        return True

    @url.validator
    def _check_url(self, attribute, value) -> bool:
        if value is None:
            pass
        elif not isinstance(value, str):
            msg = "`url` MUST be None or str."
            raise ValueError(msg)
        elif not value.endswith(".json"):
            msg = "`url` MUST end with extension '.json'"
            raise ValueError(msg)
        return True

    @metadata.validator
    def _check_metadata(self, attribute, value) -> bool:
        if not isinstance(value, dict):
            msg = "`metadata` MUST be a dict."
            raise ValueError(msg)

        needed_aliases = ["observed", "id_col_sim", "target"]
        has_needed_aliases = all((alias in value for alias in needed_aliases))
        if not has_needed_aliases:
            msg = f"`metadata` MUST contain the following:\n{needed_aliases}"
            raise ValueError(msg)

        aliases_are_all_str = all(
            (isinstance(alias, str) for alias in needed_aliases)
        )
        if not aliases_are_all_str:
            msg = (
                "Each of the following must map to a `str` in `metadata`\n"
                f"{needed_aliases}"
            )
            raise ValueError(msg)
        return True


@attr.s
class View(Protocol):
    """
    Base class for Checkrs visualizations. Provides a view of one's data.
    """

    @classmethod
    def from_chart_data(cls, data: ChartData) -> "View":
        """
        Instantiates the view from the given `ChartData`.
        """
        pass

    def draw(self, backend: str) -> ViewObject:
        """
        Renders the view of the data using a specified backend.
        """
        pass

    def save(self, filename: str) -> bool:
        """
        Saves the view of the data using the appropriate backend for the
        filename's extension. Returns True if saving succeeded.
        """
        pass
