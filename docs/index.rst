cht_tide
########

``cht_tide`` provides tidal harmonic analysis, prediction, and access to global
tide models and tide station databases for the Coastal Hazards Toolkit.

Key capabilities:

* **Harmonic analysis** -- decompose a water level time series into tidal
  constituents using the :class:`~cht_tide.tide.Tide` class.
* **Tidal prediction** -- synthesise a water level time series from constituent
  amplitudes and phases via :func:`~cht_tide.tide_predict.predict`.
* **Global tide models** -- interpolate tidal constituents from models such as
  FES2014 onto arbitrary locations using :class:`~cht_tide.model.TideModel`.
* **Tide station databases** -- query and download data from online tide station
  collections via :class:`~cht_tide.tide_stations.TideStationsDatabase`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   user_guide/index
   api/index
   changelog
