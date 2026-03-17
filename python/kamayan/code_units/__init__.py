"""Code units for configuring Kamayan simulations.

This module provides high-level Python classes for configuring various aspects
of Kamayan simulations including grid setup, physics modules, equation of state,
outputs, and runtime parameters.

Main classes:
    - KamayanGrid: Mesh and boundary configuration
    - Driver: Time integration settings
    - KamayanPhysics: Physics module selection
    - Hydro: Hydrodynamics configuration
    - GammaEos: Ideal gas equation of state
    - KamayanOutputs: Output file configuration
    - KamayanParams: Runtime parameter management

Example:
    Basic configuration setup::

        from kamayan.code_units import driver, physics
        from kamayan.code_units.Grid import UniformGrid, outflow_box
        from kamayan.code_units.Hydro import Hydro
        from kamayan.code_units.eos import GammaEos

        # Configure grid
        grid = UniformGrid(nx1=128, nx2=128, nx3=1)
        grid.boundary_conditions = outflow_box()

        # Configure physics
        phys = physics.KamayanPhysics()
        phys.hydro = Hydro(reconstruction="plm")
        phys.eos = GammaEos(gamma=1.4)

        # Configure driver
        drv = driver.Driver(tlim=0.2)
"""
