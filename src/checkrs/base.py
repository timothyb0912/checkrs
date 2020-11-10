"""
Basic objects that serve as templates for specialized functions and classes.
"""
from typing import Dict, Optional, Protocol, Union

import attr
from matplotlib.figure import Figure
from altair import Chart


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

    def from_chart_data(self, data: ChartData) -> "View":
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
