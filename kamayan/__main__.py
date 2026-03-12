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

import sys
from pathlib import Path

import typer

from kamayan.cli.app import extract_params
from kamayan.cli.utils import load_simulation


def create_script_app(script_path: Path) -> typer.Typer:
    """Create a Typer app dynamically for a simulation script."""

    # Load simulation and extract params
    try:
        sim_func = load_simulation(str(script_path))
        func = getattr(sim_func, "func", sim_func)
        params = extract_params(func)
    except AttributeError:
        typer.echo(
            f"Error: Script {script_path} must use @kamayan_app decorator",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error loading simulation: {e}", err=True)
        raise typer.Exit(1)

    # Create new Typer instance for this script with detailed help
    parthenon_args_help = """
Parthenon Arguments (pass after --):
- -r <file>    Restart from checkpoint
- -a <file>    Analyze/postprocess
- -d <dir>     Run directory
- -t <time>    Wall time limit
- -n           Dry run
- block/param=value  Override params"""
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

    # Build parameter definitions for typer
    param_defs = []
    for p in params:
        name = p["name"]
        option = p["option_name"]
        default = p["default"]
        param_type = p["annotation"]

        if param_type is bool:
            param_defs.append(f"{name}: bool = typer.Option({default})")
        else:
            type_name = (
                param_type.__name__
                if hasattr(param_type, "__name__")
                else str(param_type)
            )
            param_defs.append(
                f'{name}: {type_name} = typer.Option({repr(default)}, "{option}")'
            )

    # Join with trailing comma
    params_code = ",\n    ".join(param_defs) + ","

    # Generate run command with dynamic params
    cmd_source = f'''
def run(
    {params_code}
    dry_run: bool = typer.Option(False, "--dry-run", "-n"),
):
    """Run {script_path.name} simulation.

Parthenon Arguments (pass after --):
  -r <file>    Restart from checkpoint
  -a <file>    Analyze/postprocess
  -d <dir>     Run directory
  -t <time>    Wall time limit
  -n           Dry run
  block/param=value  Override params"""
    # Collect kwargs (filter out None)
    sim_kwargs = {{}}
    for p in params:
        val = locals().get(p["name"])
        if val is not None:
            sim_kwargs[p["name"]] = val
    
    # Run simulation
    km = sim_func(**sim_kwargs)
    
    if dry_run:
        km.write_input()
        typer.echo(f"Input file written to: {{km.input_file}}")
    else:
        km.execute()
'''

    exec_globals = {
        "typer": typer,
        "sim_func": sim_func,
        "params": params,
    }
    exec_locals = {}
    exec(cmd_source, exec_globals, exec_locals)

    # Register the run command with rich help panel
    run_func = exec_locals["run"]
    app.command("run", rich_help_panel="Parthenon Arguments (pass after --)")(run_func)

    return app


def display_help(script: Path, params: list):
    """Display help with simulation-specific parameters using Typer/Rich styling."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print(f"\n[bold cyan]Simulation:[/bold cyan] {script.name}")

    if params:
        table = Table(
            title="Simulation Parameters", show_header=True, header_style="bold magenta"
        )
        table.add_column("Option", style="cyan", no_wrap=True)
        table.add_column("Type", style="dim")
        table.add_column("Default", style="green")
        table.add_column("Description", style="white")

        for p in params:
            param_type = (
                p["annotation"].__name__
                if hasattr(p["annotation"], "__name__")
                else str(p["annotation"])
            )
            default = str(p["default"]) if p["default"] is not None else "None"
            help_text = p.get("help") or ""
            table.add_row(
                p["option_name"],
                param_type,
                default,
                help_text,
            )
        console.print(table)
    else:
        console.print("[dim]No simulation-specific parameters[/dim]")

    console.print("\n[bold cyan]Parthenon Arguments:[/bold cyan]")
    console.print("  [cyan]-r[/cyan] <file>    Restart from checkpoint file")
    console.print("  [cyan]-a[/cyan] <file>    Analyze/postprocess file")
    console.print("  [cyan]-d[/cyan] <dir>     Set run directory")
    console.print("  [cyan]-t[/cyan] <time>    Set wall time limit")
    console.print("  [cyan]-n[/cyan]           Parse input and quit (dry run)")
    console.print("  [cyan]block/param=value[/cyan]  Override input parameters")

    console.print("\n[bold cyan]Examples:[/bold cyan]")
    console.print(f"  [green]kamayan[/green] {script}  # Run simulation")
    console.print(f"  [green]kamayan[/green] {script} --dry-run  # Generate input only")
    if params:
        example_params = " ".join(
            [p["option_name"] + " " + str(p["default"]) for p in params[:2]]
        )
        console.print(
            f"  [green]kamayan[/green] {script} {example_params}  # With parameters"
        )


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
        sim_help: bool = typer.Option(
            False, "--sim-help", help="Show simulation-specific parameters"
        ),
    ):
        """Run a Kamayan simulation from a script or module.

        All arguments after the script path are parsed - simulation parameters
        (--nxb, etc.) go to the simulation, other arguments go to Parthenon.

        **Getting Help:**
        - Use 'kamayan script.py --help' to see simulation-specific parameters
        - Use 'kamayan script.py --sim-help' for simulation parameters

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
        # Handle --sim-help flag
        if sim_help:
            if script is None:
                typer.echo("Error: --sim-help requires a script path", err=True)
                raise typer.Exit(1)
            try:
                sim_func = load_simulation(str(script))
                func = getattr(sim_func, "func", sim_func)
                params = extract_params(func)
                display_help(script, params)
            except Exception as e:
                typer.echo(f"Error loading simulation: {e}", err=True)
            raise typer.Exit()
            try:
                sim_func = load_simulation(str(script))
                func = getattr(sim_func, "func", sim_func)
                params = extract_params(func)
                display_help(script, params)
            except Exception as e:
                typer.echo(f"Error loading simulation: {e}", err=True)
            raise typer.Exit()

        # Otherwise, show how to use the new syntax
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

    # Check if first arg is "run" (backward compatibility - deprecated)
    if sys.argv[1] == "run":
        typer.echo(
            "Note: 'kamayan run' is deprecated. Use 'kamayan <script.py>' instead."
        )
        # Strip "run" and use remaining args
        sys.argv = [sys.argv[0]] + sys.argv[2:]

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
