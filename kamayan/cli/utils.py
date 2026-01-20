"""Utilities for loading Kamayan simulations from scripts and modules."""

import importlib
import importlib.util
import sys
from importlib import import_module
from pathlib import Path
from typing import Callable, Optional


def load_simulation_from_script(script_path: Path) -> Callable:
    """Load a @kamayan_app decorated function from a Python script.

    Args:
        script_path: Path to the Python script file

    Returns:
        The decorated simulation function
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    if script_path.suffix != ".py":
        raise ValueError(f"Expected a .py file, got: {script_path.suffix}")

    parent_dir = script_path.parent.resolve()
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    module_name = script_path.stem
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sim_func = _find_kamayan_simulation(module)
    if sim_func is None:
        raise ValueError(
            f"No @kamayan_app decorated function found in {script_path}. "
            "Make sure your simulation function is decorated with @kamayan_app."
        )

    return sim_func


def load_simulation_from_module(module_path: str) -> Callable:
    """Load a @kamayan_app decorated function from a module path.

    Args:
        module_path: Dotted module path (e.g., 'mypackage.mysim' or 'mypackage.mysim.my_func')

    Returns:
        The decorated simulation function
    """
    if "." in module_path:
        module_path, func_name = module_path.rsplit(".", 1)
        module = import_module(module_path)
        sim_func = getattr(module, func_name, None)
        if sim_func is None:
            raise AttributeError(
                f"Function '{func_name}' not found in module '{module_path}'"
            )
    else:
        raise ValueError(
            f"Invalid module path format: {module_path}. "
            "Expected format: 'package.module.function' or './script.py'"
        )

    return sim_func


def _find_kamayan_simulation(module) -> Optional[Callable]:
    """Find a @kamayan_app decorated function in a module.

    The decorator adds an 'app' attribute to the function, which we can use
    to identify decorated functions.
    """
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name, None)
        if attr is not None and hasattr(attr, "app"):
            return attr
    return None


def load_simulation(target: str) -> Callable:
    """Load a simulation from either a script path or module path.

    Args:
        target: Either a path to a Python script (./sim.py) or
                a dotted module path (package.module.function)

    Returns:
        The decorated simulation function
    """
    path = Path(target)
    if path.exists() or path.suffix == ".py":
        return load_simulation_from_script(path)
    else:
        return load_simulation_from_module(target)
