User guide
==========

Harmonic analysis
-----------------

The :class:`~cht_tide.tide.Tide` class performs classical harmonic analysis on a
water level time series. It decomposes the signal into a set of tidal
constituents, each described by an amplitude and a phase.

.. code-block:: python

   from cht_tide.tide import Tide

   t = Tide(times=time_array, values=wl_array)
   t.analyse()

   # Access results
   t.constituent_names   # list of str
   t.amplitudes          # 1-D array (metres)
   t.phases              # 1-D array (degrees)

The analysis uses a least-squares fit with nodal corrections computed from the
:mod:`cht_tide.nodal_corrections` and :mod:`cht_tide.astro` modules.

Tidal prediction
----------------

:func:`~cht_tide.tide_predict.predict` synthesises a water level time series
from known constituent amplitudes and phases:

.. code-block:: python

   from cht_tide.tide_predict import predict

   wl = predict(times, constituent_names, amplitudes, phases)

Both ``times`` and the output are NumPy arrays. The function applies the
correct astronomical arguments and nodal corrections automatically.

Global tide models
------------------

:class:`~cht_tide.model.TideModel` provides access to gridded global tide
models (currently FES2014). It can interpolate constituent amplitudes and
phases onto arbitrary geographic coordinates:

.. code-block:: python

   from cht_tide.model import TideModel

   model = TideModel(path="/data/tide_models/fes2014")

   # Get amplitudes and phases at a list of points
   amps, phases = model.get_constituents(lon=[4.0, 5.0], lat=[52.0, 53.0])

The :class:`~cht_tide.database.TideModelDatabase` manages a collection of
tide models (local and remote via S3) and provides a higher-level interface:

.. code-block:: python

   from cht_tide.database import TideModelDatabase

   db = TideModelDatabase(path="/data/tide_models")
   db.read()

   # List available models
   print(db.dataset_names)

Tide station databases
----------------------

:class:`~cht_tide.tide_stations.TideStationsDatabase` manages collections of
tide station datasets that contain harmonic constituent information at discrete
station locations:

.. code-block:: python

   from cht_tide.tide_stations import TideStationsDatabase

   db = TideStationsDatabase(path="/data/tide_stations")
   db.read()

   # Get a GeoDataFrame of all stations in a dataset
   gdf = db.datasets["xtide"].gdf

   # Get constituents at a specific station
   station = db.datasets["xtide"].get_station(name="Sewells Point")
