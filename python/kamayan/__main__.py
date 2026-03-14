"""Kamayan main CLI entry point.

Usage:
    kamayan <script.py> [options]
    kamayan run <script.py> [options]

Examples:
    kamayan ./my_simulation.py
    kamayan ./my_simulation.py --dry-run
    kamayan ./my_simulation.py --nxb 64
    kamayan run ./my_simulation.py --dry-run
"""

import functools
import inspect
import sys
from pathlib import Path
from typing import Optional

import typer

from kamayan.cli.utils import load_simulation
from kamayan.cli.app import KamayanSimulation


def create_main_help_app() -> typer.Typer:
    """Create the main help app shown when no script is provided."""
    app = typer.Typer(
        help="Kamayan simulation manager",
        rich_markup_mode="markdown",
        no_args_is_help=True,
    )

    @app.command("run", context_settings={"allow_interspersed_args": False})
    def run_command(
        ctx: typer.Context,
        script: Path | None = typer.Argument(
            None,
            help="Python script path (e.g., ./sim.py)",
        ),
    ):
        """Run a Kamayan simulation from a script or module.

        All arguments after the script path are parsed - simulation parameters
        (--nxb, etc.) go to the simulation, other arguments go to Parthenon.

        **Getting Help:**
        - Use 'kamayan script.py --help' to see simulation-specific parameters

        **Parthenon Arguments:**
        - `-r <file>` - Restart from checkpoint file
        - `-a <file>` - Analyze/postprocess file
        - `-d <directory>` - Set run directory
        - `-t hh:mm:ss` - Set wall time limit
        - `-n` - Parse input and quit (dry run)
        - `block/param=value` - Override input parameters

        **Examples:**

        ```bash
        # Basic run
        kamayan ./my_sim.py

        # Show simulation parameters
        kamayan ./my_sim.py --help

        # Dry run
        kamayan ./my_sim.py --dry-run

        # Override simulation parameters
        kamayan ./my_sim.py --nxb 64

        # Mix simulation params and parthenon args
        kamayan ./my_sim.py --nxb 64 parthenon/time/nlim=100 -r restart.rhdf
        ```
        """
        typer.echo(f"Running: {script}")
        typer.echo(
            "Note: For this script, use 'kamayan {script} --help' to see available parameters."
        )
        raise typer.Exit()

    return app


def _show_help(err: Optional[Exception] = None):
    """Display main help and exit."""
    original_argv = sys.argv
    sys.argv = ["kamayan", "--help"]
    main_app = create_main_help_app()
    try:
        main_app()
    finally:
        sys.argv = original_argv
        if err:
            typer.secho(f"ERROR:\n{err}", fg=typer.colors.RED, bold=True, err=True)
            raise err


def main():
    """Entry point for the kamayan CLI."""
    # Show main help when:
    # - No args provided (kamayan)
    # - --help or -h passed (kamayan --help)
    # - First arg is not a valid script path
    show_help = False

    if len(sys.argv) < 2:
        show_help = True
    elif sys.argv[1] in ("--help", "-h"):
        show_help = True

    if show_help:
        _show_help()
        return

    script_target = sys.argv[1]
    try:
        sim_func = load_simulation(script_target)
    except Exception as e:
        _show_help(e)
        return

    # Create dynamic app for this script
    script_app = KamayanSimulation(sim_func).app

    # Replace sys.argv so the new app sees only its args
    sys.argv = sys.argv[1:]  # Remove "kamayan", keep script + args
    script_app()


if __name__ == "__main__":
    main()
