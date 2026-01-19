"""Kamayan main CLI entry point.

Usage:
    kamayan run <script_or_module> [--dry-run] [parthenon_args...]

Examples:
    kamayan run ./my_simulation.py
    kamayan run ./my_simulation.py --dry-run
    kamayan run ./my_simulation.py -r restart.dat
    kamayan run my_package.my_simulation
"""

from pathlib import Path
from typing import TYPE_CHECKING

import typer

from kamayan.cli.utils import load_simulation

if TYPE_CHECKING:
    pass


app = typer.Typer(
    help="Kamayan simulation manager",
    rich_markup_mode="markdown",
    no_args_is_help=True,
)


@app.command("run")
def run_command(
    script: Path = typer.Argument(
        ...,
        help="Python script path (e.g., ./sim.py) or module path (e.g., mypackage.mysim)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Generate input file without executing"
    ),
):
    """Run a Kamayan simulation from a script or module.

    Any additional arguments are forwarded to Parthenon.

    Parthenon arguments:
        -r <file>       Restart with this file
        -a <file>       Analyze/postprocess this file
        -d <directory>  Specify run directory
        -t hh:mm:ss     Wall time limit
        -n              Parse input file and quit
        -m <nproc>      Output mesh structure and quit
        -c              Show configuration and quit
        -h              Show help
        block/par=value Override parameter values

    Examples:
        kamayan run ./my_simulation.py
        kamayan run ./my_simulation.py --dry-run
        kamayan run ./my_simulation.py -r restart.dat
        kamayan run ./my_simulation.py -d /tmp/run
        kamayan run my_package.my_simulation
    """
    try:
        sim_func = load_simulation(str(script))
        km = sim_func()
    except Exception as e:
        typer.echo(f"Error loading simulation: {e}", err=True)
        raise typer.Exit(1)

    if dry_run:
        km.write_input()
        typer.echo(f"Input file written to: {km.input_file}")
    else:
        km.execute()


@app.command("version")
def version():
    """Show Kamayan version information."""
    try:
        from kamayan import __version__

        typer.echo(f"Kamayan: {__version__}")
    except ImportError:
        typer.echo("Kamayan: unknown")


def main():
    """Entry point for the kamayan CLI."""
    app()


if __name__ == "__main__":
    app()
