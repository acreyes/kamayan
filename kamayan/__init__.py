"""Kamayan: A flexible framework for astrophysical hydrodynamics simulations.

Kamayan provides both C++ and Python interfaces for running MHD simulations
built on the Parthenon AMR framework. The Python interface offers code_units
for configuring simulations and a CLI system for running them.

Key components:
    - KamayanManager: Main simulation configuration and execution manager
    - kamayan_app: Decorator for creating CLI applications from simulations
    - code_units: Grid, Driver, Physics, Hydro, EOS configuration classes

Example:
    Basic simulation setup::

        from kamayan import KamayanManager, process_units
        from kamayan.code_units.Grid import UniformGrid

        # Configure and run simulation
        uc = process_units("my_sim", setup_params=setup_params)
        km = KamayanManager(uc, name="simulation")
        km.execute()

    Using the CLI decorator::

        from kamayan import kamayan_app

        @kamayan_app(description="My simulation")
        def my_simulation():
            # Configure your simulation
            return km

Version: 0.1.0
"""
