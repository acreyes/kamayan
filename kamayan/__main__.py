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

import typer

from kamayan.cli.utils import load_simulation
from kamayan.cli.app import KamayanSimulation
from kamayan_manager import KamayanManager


def create_script_app(script_path: Path) -> typer.Typer:
    """Create a Typer app dynamically for a simulation script."""
    try:
        sim_func = load_simulation(str(script_path))
        func = getattr(sim_func, "func", sim_func)
    except AttributeError:
        typer.echo(
            f"Error: Script {script_path} must use @kamayan_app decorator",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error loading simulation: {e}", err=True)
        raise typer.Exit(1)

    return KamayanSimulation(sim_func).app

    # Create new Typer instance for this script with detailed help
    parthenon_args_help = """
**Parthenon Arguments:**
- `-r <file>` - Restart from checkpoint file
- `-a <file>` - Analyze/postprocess file
- `-d <directory>` - Set run directory
- `-t hh:mm:ss` - Set wall time limit
- `-n` - Parse input and quit (dry run)
- `block/param=value` - Override input parameters
"""
    app = typer.Typer(
        help=f"""
Kamayan simulation: {script_path.name}

Run {script_path.name} with optional parameters below.
{parthenon_args_help}

Examples:
  kamayan {script_path.name} --dry-run
  kamayan {script_path.name} --nxb 64
  kamayan {script_path.name} -- parthenon/time/nlim=100 -r restart.rhdf""",
        rich_markup_mode="markdown",
    )

    @functools.wraps(sim_func)
    def run_kamayan(*args, **kwargs):
        ctx: typer.Context = kwargs.pop("ctx")
        try:
            km: KamayanManager = sim_func(*args, **kwargs)
        except Exception as e:
            typer.echo(f"Error loading simulation: {e}", err=True)
            raise typer.Exit(1)

        km.execute(*ctx.args)

    # Force Typer to see the user_func's signature
    user_signature = inspect.signature(sim_func)
    ctx_param = inspect.Parameter(
        "ctx", inspect.Parameter.KEYWORD_ONLY, annotation=typer.Context
    )

    # Build the new signature (User Args + Context)
    new_params = list(user_signature.parameters.values()) + [ctx_param]
    new_signature = user_signature.replace(parameters=new_params)
    setattr(run_kamayan, "__signature__", new_signature)

    # Copy annotations so Typer knows the types for CLI casting
    new_annotations = dict(sim_func.__annotations__)
    new_annotations["ctx"] = typer.Context
    run_kamayan.__annotations__ = new_annotations

    # combine our docstrings
    user_doc = sim_func.__doc__ or ""
    run_kamayan.__doc__ = f"{user_doc}\n\n{parthenon_args_help}"

    app.command(
        "run",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )(run_kamayan)

    return app


# Main entry point - decides between new dynamic app or backward compat
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
    elif not Path(sys.argv[1]).exists():
        show_help = True

    if show_help:
        # Show help explicitly
        original_argv = sys.argv
        sys.argv = ["kamayan", "--help"]
        main_app = create_main_help_app()
        try:
            main_app()
        finally:
            sys.argv = original_argv
        return

    # Check if first arg is a script path
    script_path = Path(sys.argv[1])

    if not script_path.exists():
        typer.echo(f"Error: Script not found: {script_path}", err=True)
        raise typer.Exit(1)

    # Create dynamic app for this script
    script_app = create_script_app(script_path)

    # Replace sys.argv so the new app sees only its args
    # sys.argv[0] = script_path.name  # Optional: change argv[0] to script name
    sys.argv = sys.argv[1:]  # Remove "kamayan", keep script + args

    # Run the script-specific app
    script_app()


if __name__ == "__main__":
    main()
