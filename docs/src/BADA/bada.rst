BADA Methods
============

Support for the `Base of Aircraft Data <https://www.eurocontrol.int/model/bada>`_ revision 3.0 (BADA-3) and custom performance
data structured in the same way is included in ``AEIC``. Below are the various dataclasses, methods, and helper functions
that can be used to evaluate or manipulate BADA-3 formatted data.

Engine and Fuel Burn Models
---------------------------

.. autoclass:: src.BADA.model.Bada3EngineModel
    :members:

.. autoclass:: src.BADA.model.Bada3JetEngineModel
    :members:

.. autoclass:: src.BADA.model.Bada3TurbopropEngineModel
    :members:

.. autoclass:: src.BADA.model.Bada3PistonEngineModel
    :members:

.. autoclass:: src.BADA.model.Bada3FuelBurnModel
    :members:


Aircraft Parameters
-------------------

.. autoclass:: src.BADA.aircraft_parameters.Bada3AircraftParameters
    :members:

Fuel Burn Base Classes
----------------------

.. autoclass:: src.BADA.fuel_burn_base.BaseAircraftParameters
    :members:

.. autoclass:: src.BADA.fuel_burn_base.BaseFuelBurnModel
    :members:

Helper Functions
----------------

.. automodule:: src.BADA.helper_functions
    :members: