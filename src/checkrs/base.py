"""
Basic objects that serve as templates for specialized functions and classes.
"""
from typing import (
    Dict, Optional, Union
)
from typing_extensions import Protocol

import attr
import numpy as np
import pandas as pd
import torch
from plotnine import ggplot
from altair import TopLevelMixin

from checkrs.utils import progress


EXTENSIONS_PLOTNINE = set((".png", ".pdf"))
EXTENSIONS_ALTAIR = set((".html", ".json"))
EXTENSIONS = EXTENSIONS_PLOTNINE | EXTENSIONS_ALTAIR

ViewObject = Union[ggplot, TopLevelMixin]
TensOrArray = Union[np.ndarray, torch.Tensor]


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
        following keys: {"observed", "id_sim", "target"}. These should map
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
        id_col_sim = self.metadata["id_sim"]
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
        id_col_sim = self.metadata["id_sim"]
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

        needed_aliases = ["observed", "id_sim", "target"]
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

    @classmethod
    def _get_id_sim(cls, tidy_df: pd.DataFrame) -> int:
        if tidy_df.shape[0] == 0:
            return 0
        else:
            return int(tidy_df["id_sim"].max())

    @classmethod
    def _add_dataset_to_tidy_df(
        cls,
        tidy_df: pd.DataFrame,
        new_y: np.ndarray,
        x: Optional[pd.DataFrame]=None,
        observed: bool=False
    ) -> pd.DataFrame:
        to_add = pd.DataFrame({
            "target": new_y,
        })
        to_add["observed"] = observed
        to_add["id_sim"] = cls._get_id_sim(tidy_df) + 1

        if x is not None:
            to_add = to_add.join(x)

        return tidy_df.append(to_add, ignore_index=True)

    @classmethod
    def _make_tidy_df_from_raw(
        cls,
        targets: TensOrArray,
        targets_simulated: TensOrArray,
        design: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        targets_np = (
            targets if isinstance(targets, np.ndarray) else targets.numpy()
        )
        targets_simulated_np = (
            targets_simulated if isinstance(targets_simulated, np.ndarray)
            else targets_simulated.numpy()
        )

        tidy_df = pd.DataFrame(columns=["target", "observed", "id_sim"])
        tidy_df = cls._add_dataset_to_tidy_df(
            tidy_df, new_y=targets_np, x=design, observed=True
        )
        for col in progress(range(targets_simulated_np.shape[1])):
            tidy_df = cls._add_dataset_to_tidy_df(
                tidy_df,
                new_y=targets_simulated_np[:, col],
                x=design,
                observed=False
            )
        tidy_df.id_sim = tidy_df.id_sim.astype(int)
        return tidy_df

    @classmethod
    def from_raw(
        cls,
        targets: TensOrArray,
        targets_simulated: TensOrArray,
        design: Optional[pd.DataFrame],
        url: Optional[str] = None,
    ) -> "ChartData":
        """
        Instantiates a ChartData object from the raw data.

        Parameters
        ----------
        targets : Union[torch.Tensor, np.ndarray]
            1D Tensor or 1D ndarray. Denotes the target variables for the rows
            in `design`.
        targets_simulated : Union[torch.Tensor, np.ndarray]
            2D Tensor or 2D ndarray. Each column denotes a simulated set of
            target values for the rows in `design`.
        url : optional, str or None.
            Denotes the location that the data is currently saved at, as a json
            file, or where the data's json should be saved. Should have a
            '.json' file extension. Default is None.
        design : optional, pd.DataFrame or None.
            If DataFrame, this should be the design matrix used to create
            `targets_simulated` and any additional variables of interest that
            can be associated with each row of the original design matrix.
            Default is None.

        Returns
        -------
        Instantiated ChartData object.
        """
        tidy_df = cls._make_tidy_df_from_raw(
            targets, targets_simulated, design
        )

        metadata = {
            "target": "target",
            "observed": "observed",
            "id_sim": "id_sim",
        }
        return cls(data=tidy_df, url=url, metadata=metadata)


@attr.s
class PlotTheme:
    """
    Default attributes for a plot
    """
    label_y : str = attr.ib()
    plotting_col : str = attr.ib(default="target")
    _label_x : Optional[str] = attr.ib(default=None)
    title : Optional[str] = attr.ib(default=None)
    rotation_y : int = attr.ib(default=0)
    rotation_x_ticks : int = attr.ib(default=0)
    padding_y_plotnine : int = attr.ib(default=40)
    padding_y_altair : int = attr.ib(default=100)
    dpi_print : int = attr.ib(default=500)
    dpi_web : int = attr.ib(default=72)
    fontsize : int = attr.ib(default=13)
    color_observed : str = attr.ib(default="#045a8d")
    color_simulated : str = attr.ib(default="#a6bddb")
    width_inches : int = attr.ib(default=5)
    height_inches : int = attr.ib(default=3)

    @property
    def label_x(self) -> str:
        label = (
            self._label_x if self._label_x is not None else self.plotting_col
        )
        return label

    @property
    def width_pixels(self) -> int:
        return self.dpi_web * self.width_inches

    @property
    def height_pixels(self) -> int:
        return self.dpi_web * self.height_inches


@attr.s
class View(Protocol):
    """
    Base class for Checkrs visualizations. Provides a view of one's data.
    """
    data: ChartData
    theme: PlotTheme

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
