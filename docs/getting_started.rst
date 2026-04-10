Getting started
===============

Installation
------------

Install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/Deltares-research/cht_tide.git

For development:

.. code-block:: bash

   git clone https://github.com/Deltares-research/cht_tide.git
   cd cht_tide
   pip install -e ".[tests]"

Quick example -- predict tides
-------------------------------

Generate a tidal prediction at a single location using constituent amplitudes
and phases from a global tide model:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from cht_tide.tide_predict import predict

   # Time axis: 7 days at 10-minute intervals
   times = pd.date_range("2024-01-01", periods=7 * 144, freq="10min")

   # Define a few constituents (amplitude in metres, phase in degrees)
   constituents = ["M2", "S2", "N2", "K1", "O1"]
   amplitudes = np.array([0.85, 0.22, 0.18, 0.15, 0.12])
   phases = np.array([120.0, 145.0, 100.0, 30.0, 350.0])

   # Predict water levels
   wl = predict(times, constituents, amplitudes, phases)

   print(f"Max water level: {wl.max():.2f} m")
   print(f"Min water level: {wl.min():.2f} m")

Quick example -- harmonic analysis
-----------------------------------

Decompose an observed water level time series into tidal constituents:

.. code-block:: python

   from cht_tide.tide import Tide

   # Assume `df` is a DataFrame with a DatetimeIndex and column "wl"
   t = Tide(times=df.index, values=df["wl"].values)
   t.analyse()

   # Print amplitudes and phases
   for name, amp, phase in zip(t.constituent_names, t.amplitudes, t.phases):
       print(f"{name:>4s}  amp={amp:.3f} m  phase={phase:.1f} deg")
