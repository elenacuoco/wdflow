API reference
=============

``wdflow`` has two layers: trigger generation (``wdf.config``, ``wdf.processes``,
``wdf.observers``, ``wdf.structures`` -- needs the compiled ``pytsa``/p4TSA core) and downstream
trigger analysis (``wdf.analysis`` -- clustering, multi-detector coincidence,
background/false-alarm-probability, and ROC analysis; works standalone on already-saved trigger
files, no ``pytsa`` required). Each module below is documented from its own docstrings.

Trigger generation
-------------------

wdf.config.Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.config.Parameters
   :members:

wdf.processes.wdfUnitDSWorker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.processes.wdfUnitDSWorker
   :members:

wdf.processes.wdf
~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.processes.wdf
   :members:

wdf.processes.BandPassDownSampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.processes.BandPassDownSampling
   :members:

wdf.processes.zero_phase_whitening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.processes.zero_phase_whitening
   :members:

wdf.processes.DWhitening
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.processes.DWhitening
   :members:

wdf.processes.Whitening
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.processes.Whitening
   :members:

wdf.processes.wavelet_energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.processes.wavelet_energy
   :members:

wdf.observers.ParameterEstimationObserver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.observers.ParameterEstimationObserver
   :members:

wdf.observers.SingleEventPrintFileObserver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.observers.SingleEventPrintFileObserver
   :members:

wdf.structures.eventPE
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.structures.eventPE
   :members:

Downstream analysis
---------------------

wdf.analysis.io
~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.io
   :members:

wdf.analysis.coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.coefficients
   :members:

wdf.analysis.metaparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.metaparameters
   :members:

wdf.analysis.output_schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.output_schema
   :members:

wdf.analysis.cluster_coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.cluster_coefficients
   :members:

wdf.analysis.clustering
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.clustering
   :members:

wdf.analysis.coincidence
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.coincidence
   :members:

wdf.analysis.significance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.significance
   :members:

wdf.analysis.evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.evaluation
   :members:

wdf.analysis.roc
~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.roc
   :members:

wdf.analysis.pairs
~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.pairs
   :members:

wdf.analysis.scale
~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.scale
   :members:

wdf.analysis.pixel_graph
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.pixel_graph
   :members:

wdf.analysis.detector_graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.detector_graph
   :members:

wdf.analysis.network_graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.network_graph
   :members:

wdf.analysis.baseline
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.baseline
   :members:

wdf.analysis.gnn
~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.gnn
   :members:

wdf.analysis.wavelets
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.wavelets
   :members:

wdf.analysis.reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.reconstruction
   :members:

wdf.analysis.robust_events
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.robust_events
   :members:

wdf.analysis.injections
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.injections
   :members:

wdf.analysis.plots
~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.plots
   :members:

wdf.analysis.report
~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.report
   :members:

wdf.analysis.review_report
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.analysis.review_report
   :members:

Simulated data
---------------
wdf.mock.waveforms
~~~~~~~~~~~~~~~~~~

.. automodule:: wdf.mock.waveforms
   :members:

wdf.mock.dataset
~~~~~~~~~~~~~~~~

.. automodule:: wdf.mock.dataset
   :members:
