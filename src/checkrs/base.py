"""
Basic objects that serve as templates for specialized functions and classes.
"""
from typing import (
    Dict, Optional, Union
)
from typing_extensions import Protocol

import attr
import pandas as pd
from matplotlib.figure import Figure
from altair import Chart


EXTENSIONS_PLOTNINE = set((".png", ".pdf"))
EXTENSIONS_ALTAIR = set((".html", ".json"))
EXTENSIONS = EXTENSIONS_PLOTNINE | EXTENSIONS_ALTAIR

ViewObject = Union[Figure, Chart]


@attr.s
class ChartData:
    data: Optional[pd.DataFrame] = attr.ib()
    url: Optional[str] = attr.ib()
    metadata: Dict[str, str] = attr.ib()


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
