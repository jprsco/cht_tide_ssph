"""cht_tide — tidal harmonic analysis and prediction library.

Exposes the main public API: database access, model classes, prediction,
and tide station lookups.
"""

__version__ = "0.1.1"

from cht_tide.database import TideModelDatabase
from cht_tide.model import TideModel
from cht_tide.tide_predict import predict
from cht_tide.tide_stations import TideStationsDatabase

__all__ = ["TideModelDatabase", "TideModel", "predict", "TideStationsDatabase"]
