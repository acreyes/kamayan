"""Kamayan main CLI entry point.

Usage:
    kamayan <script_or_module> [--dry-run] [simulation_args...] [parthenon_args...]

Examples:
    kamayan ./my_simulation.py
    kamayan ./my_simulation.py --dry-run
    kamayan ./my_simulation.py -r restart.dat
    kamayan ./my_simulation.py parthenon/time/nlim=100
    kamayan ./my_simulation.py --nxb 64 --nblocks 8 parthenon/time/nlim=100
"""

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from kamayan.cli.app import extract_params
from kamayan.cli.utils import load_simulation

if TYPE_CHECKING:
    pass


app = typer.Typer(
    help="Kamayan simulation manager",
    rich_markup_mode="markdown",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def parse_arguments(
    args: list[str], params: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Parse arguments into simulation kwargs and parthenon args."""
    param_names = {p["name"] for p in params}
    param_options = {p["option_name"] for p in params}

    sim_kwargs = {}
    parthenon_args = []

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--dry-run", "-n"):
            parthenon_args.append(arg)
            i += 1
            continue

        normalized = arg.lstrip("-").replace("-", "_")
        option_normalized = arg.lstrip("-")

        is_sim_param = (
            arg in param_options
            or normalized in param_names
            or option_normalized in {p.lstrip("-") for p in param_options}
        )

        if is_sim_param:
            if "=" in arg:
                key, value = arg.split("=", 1)
                key = key.lstrip("-").replace("-", "_")
                sim_kwargs[key] = ast.literal_eval(value)
            else:
                key = arg.lstrip("-").replace("-", "_")
                if i + 1 < len(args):
                    sim_kwargs[key] = ast.literal_eval(args[i + 1])
                    i += 1
        else:
            parthenon_args.append(arg)

        i += 1

    return sim_kwargs, parthenon_args


def display_help(script: Path, params: list[dict[str, Any]]):
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


@app.command("run")
def run_command(
    ctx: typer.Context,
    script: Path = typer.Argument(
        ...,
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
    - Use 'kamayan script.py --sim-help' to see simulation-specific parameters
    - Use 'kamayan script.py -- --help' to pass --help to Parthenon

    **Examples:**

    ```bash
    # Show simulation parameters
    kamayan ./my_sim.py --sim-help

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
        try:
            sim_func = load_simulation(str(script))
            func = getattr(sim_func, "func", sim_func)
            params = extract_params(func)
            display_help(script, params)
        except Exception as e:
            typer.echo(f"Error loading simulation: {e}", err=True)
        raise typer.Exit()

    # Get extra args from Click context
    remaining = list(ctx.args)

    dry_run = "--dry-run" in remaining or "-n" in remaining
    remaining = [r for r in remaining if r not in ("--dry-run", "-n")]

    try:
        sim_func = load_simulation(str(script))
        func = getattr(sim_func, "func", sim_func)
        params = extract_params(func)
        sim_kwargs, parthenon_args = parse_arguments(remaining, params)

        km = sim_func(**sim_kwargs)
    except Exception as e:
        typer.echo(f"Error loading simulation: {e}", err=True)
        raise typer.Exit(1)

    if dry_run:
        km.write_input()
        typer.echo(f"Input file written to: {km.input_file}")
    else:
        km.execute(*parthenon_args)


def main():
    """Entry point for the kamayan CLI."""
    app()


if __name__ == "__main__":
    app()
