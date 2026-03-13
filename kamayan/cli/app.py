"""Kamayan CLI application with @kamayan_app decorator."""

from functools import wraps
import functools
import inspect
from typing import Annotated, Any, Callable, Optional, TYPE_CHECKING, cast

import click
import typer

if TYPE_CHECKING:
    from kamayan.kamayan_manager import KamayanManager


class KamayanSimulation:
    """Wrapper around a simulation function providing CLI commands."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the KamayanSimulation wrapper.

        Args:
            func: The simulation function to wrap
            name: Optional name for the simulation (defaults to function name)
            description: Optional description for the CLI help
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description
        self._app = typer.Typer(
            help=self.description or f"Kamayan simulation: {self.name}",
            rich_markup_mode="markdown",
        )
        self._register_commands()

    def _register_commands(self):
        func_signature = inspect.signature(self.func)
        func_params = list(func_signature.parameters.values())
        func_doc = self.func.__doc__ or ""
        # 1. Map all inputs to their parameter names and fill defaults
        bound = func_signature.bind()
        bound.apply_defaults()

        # 2. Clean the arguments: swap Typer objects for their real defaults
        for name, value in bound.arguments.items():
            if isinstance(value, (typer.models.OptionInfo, typer.models.ArgumentInfo)):
                # Replace the metadata object with its real default value (e.g., 32)
                bound.arguments[name] = value.default
        default_km: KamayanManager = self.func(**bound.arguments)

        @functools.wraps(self.func)
        def run(
            *args,
            input_file: Optional[str] = None,
            dry_run: bool = False,
            info: bool = False,
            **kwargs,
        ):
            parthenon_args: typer.Context = kwargs.pop("ctx")
            breakpoint()

            km = self.func(*args, **kwargs)
            if input_file:
                km.input_file = input_file

            if dry_run:
                km.write_input()
                typer.echo(f"Input file written to: {km.input_file}")
            elif info:
                typer.echo(f"Simulation: {km.name}")
                typer.echo(f"Input file: {km.input_file}")
                typer.echo(f"Grid: {type(km.grid).__name__}")
                typer.echo(f"Driver: {km.driver.integrator}")
                typer.echo(
                    f"Hydro: {km.physics.hydro.reconstruction} / {km.physics.hydro.riemann}"
                )
                typer.echo(f"EOS: {type(km.physics.eos).__name__}")
            else:
                km.execute(*parthenon_args.args)

        parthenon_args_help = """Run the simulation.

        Additional arguments are forwarded directly to Parthenon.

        **Parthenon Arguments:**

        - `-r <file>` - Restart from checkpoint file
        - `-a <file>` - Analyze/postprocess file
        - `-d <directory>` - Set run directory
        - `-t hh:mm:ss` - Set wall time limit
        - `-n` - Parse input and quit (dry run)
        - `-m <nproc>` - Output mesh structure and quit
        - `-c` - Show configuration and quit
        - `-h` - Show Parthenon help
        - `block/param=value` - Override input parameters
        """

        # compose our app signature with the func signature
        dry_run_param = inspect.Parameter(
            "dry_run",
            inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(
                False,
                "--dry-run",
                "-n",
                is_flag=True,
                help="Generate input file without executing simulation",
            ),
            annotation=bool,
        )
        func_params += [dry_run_param]

        input_param = inspect.Parameter(
            "input_file",
            inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(
                None,
                "--input-file",
                "-f",
                help=f"Write input file to declared file ({default_km.input_file}).",
            ),
            annotation=str,
        )
        func_params += [input_param]

        info_param = inspect.Parameter(
            "info",
            inspect.Parameter.KEYWORD_ONLY,
            default=typer.Option(
                False,
                "--info",
                "-i",
                is_flag=True,
                help="Display simulation configuration.",
            ),
            annotation=bool,
        )
        func_params += [info_param]

        ctx_param = inspect.Parameter(
            "ctx", inspect.Parameter.KEYWORD_ONLY, annotation=typer.Context
        )
        func_params += [ctx_param]

        func_signature = func_signature.replace(parameters=func_params)
        setattr(run, "__signature__", func_signature)
        run.__doc__ = f"{func_doc}\n\n{parthenon_args_help}"

        self._app.command(
            "run",
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        )(run)

    @property
    def app(self):
        """Get the Typer app instance for this simulation."""
        return self._app


def kamayan_app(
    name: Optional[str] = None,
    *,
    description: Optional[str] = None,
) -> Callable:
    """Decorator to create a Kamayan simulation CLI.

    Args:
        name: Optional name for the simulation
        description: Optional description for the CLI help

    Usage:
        @kamayan_app
        def my_simulation() -> KamayanManager:
            km = KamayanManager(...)
            # configure ...
            return km

        if __name__ == "__main__":
            my_simulation.app()

    The decorated function can also be used directly:
        km = my_simulation()
    """

    def decorator(func: Callable) -> "KamayanSimulation":
        sim = KamayanSimulation(func, name, description)

        @wraps(func)
        def wrapper(*args, **kwargs) -> "KamayanManager":
            return func(*args, **kwargs)

        cast(Any, wrapper).app = sim.app
        return cast("KamayanSimulation", wrapper)

    return decorator
