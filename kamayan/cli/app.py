"""Kamayan CLI application with @kamayan_app decorator."""

from functools import wraps
from typing import Callable, Optional, TYPE_CHECKING

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
        self.func = func
        self.name = name or func.__name__
        self.description = description
        self._app = typer.Typer(
            help=self.description or f"Kamayan simulation: {self.name}",
            rich_markup_mode="markdown",
        )
        self._register_commands()

    def _register_commands(self):
        @self._app.command("run")
        def run(
            dry_run: bool = typer.Option(
                False,
                "--dry-run",
                "-n",
                help="Generate input file without executing simulation",
            ),
        ):
            """Run the simulation."""
            km = self.func()
            if dry_run:
                km.write_input()
                typer.echo(f"Input file written to: {km.input_file}")
            else:
                km.execute()

        @self._app.command("generate-input")
        def generate_input(
            output: Optional[str] = typer.Option(
                None, "-o", "--output", help="Output file path (default: {name}.in)"
            ),
        ):
            """Generate the Parthenon input file."""
            km = self.func()
            if output:
                km.write_input(file=output)
                typer.echo(f"Input file written to: {output}")
            else:
                km.write_input()
                typer.echo(f"Input file written to: {km.input_file}")

        @self._app.command("info")
        def info():
            """Display simulation configuration."""
            km = self.func()
            typer.echo(f"Simulation: {km.name}")
            typer.echo(f"Input file: {km.input_file}")
            typer.echo(f"Grid: {type(km.grid).__name__}")
            typer.echo(f"Driver: {km.driver.integrator}")
            typer.echo(
                f"Hydro: {km.physics.hydro.reconstruction} / {km.physics.hydro.riemann}"
            )
            typer.echo(f"EOS: {type(km.physics.eos).__name__}")

        @self._app.command("version")
        def version():
            """Show version information."""
            try:
                from kamayan import __version__

                typer.echo(f"Kamayan: {__version__}")
            except ImportError:
                typer.echo("Kamayan: unknown")

    def __call__(self) -> "KamayanManager":
        return self.func()

    @property
    def app(self):
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
        def wrapper() -> "KamayanManager":
            return func()

        wrapper.app = sim.app
        return wrapper

    return decorator
