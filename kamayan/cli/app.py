"""Kamayan CLI application with @kamayan_app decorator."""

import inspect
import re
from functools import wraps
from typing import (
    Any,
    Callable,
    Optional,
    TYPE_CHECKING,
    Union,
    cast,
    get_args,
    get_origin,
)

import typer
from typing_extensions import Annotated, get_type_hints


def _param_to_option(name: str) -> str:
    """Convert parameter name to CLI option name (e.g., max_level → max-level)."""
    return re.sub(r"_", "-", name)


def _get_param_info(param: inspect.Parameter) -> dict[str, Any]:
    """Extract CLI parameter info from function parameter."""
    name = param.name
    default = param.default if param.default is not inspect.Parameter.empty else None
    annotation = (
        param.annotation if param.annotation is not inspect.Parameter.empty else Any
    )

    help_text = None
    origin = get_origin(annotation)
    if origin is Annotated:
        annotated_args = get_args(annotation)
        if annotated_args:
            help_text = annotated_args[-1]
            annotation = annotated_args[0]

    return {
        "name": name,
        "option_name": f"--{_param_to_option(name)}",
        "default": default,
        "annotation": annotation,
        "has_default": default is not None,
        "help": help_text,
    }


def _extract_params(func: Callable) -> list[dict[str, Any]]:
    """Extract CLI parameter info from function signature."""
    sig = inspect.signature(func)
    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        params.append(_get_param_info(param))
    return params


def extract_params(func: Callable) -> list[dict[str, Any]]:
    """Extract CLI parameter info from function signature.

    This is the public version of _extract_params, intended for use
    by external tools like the kamayan CLI helper.

    Args:
        func: The function to extract parameters from

    Returns:
        List of parameter info dictionaries
    """
    return _extract_params(func)


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
        self._params = _extract_params(func)
        self._app = typer.Typer(
            help=self.description or f"Kamayan simulation: {self.name}",
            rich_markup_mode="markdown",
        )
        self._register_commands()

    def _build_param_args(self) -> str:
        """Build function parameter strings for dynamic CLI options."""
        param_defs = []
        for p in self._params:
            name = p["name"]
            default = p["default"]
            annotation = p["annotation"]
            help_text = p.get("help")
            option_name = p["option_name"]

            if annotation is bool:
                param_defs.append(
                    f'{name}: bool = typer.Option({default}, "{option_name}"'
                    + (f', help="{help_text}"' if help_text else "")
                    + ")"
                )
            elif p["has_default"]:
                param_defs.append(
                    f'{name}: {annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)} = typer.Option({repr(default)}, "{option_name}"'
                    + (f', help="{help_text}"' if help_text else "")
                    + ")"
                )
            else:
                param_defs.append(
                    f'{name}: {annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)} = typer.Option(None, "{option_name}"'
                    + (f', help="{help_text}"' if help_text else "")
                    + ")"
                )
        if param_defs:
            return ",\n            ".join(param_defs) + ","
        return ""

    def _register_commands(self):
        param_args = self._build_param_args()
        params_dict = {p["name"]: p for p in self._params}

        exec_globals = {"typer": typer, "Optional": Optional, "list": list}
        exec_locals = {}

        run_source = (
            """
@self._app.command("run")
def run(
            """
            + param_args
            + '''
            dry_run: bool = typer.Option(
                False,
                "--dry-run",
                "-n",
                help="Generate input file without executing simulation",
            ),
            parthenon_args: list[str] = typer.Argument(
                None,
                help="Arguments forwarded to Parthenon (e.g., `parthenon/time/nlim=100`, `-r restart.rhdf`)",
            ),
        ):
    """Run the simulation.

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

    **Examples:**

    ```bash
    # Basic run
    python my_sim.py run

    # Override number of cycles
    python my_sim.py run parthenon/time/nlim=100

    # Restart from checkpoint
    python my_sim.py run -r output.00050.rhdf

    # Multiple parameter overrides
    python my_sim.py run parthenon/time/nlim=50 parthenon/time/tlim=0.5
    ```
    """
    sim_kwargs = {{k: v for k, v in locals().items() if k in params_dict}}
    km = self.func(**sim_kwargs)

    if dry_run:
        km.write_input()
        typer.echo(f"Input file written to: {{km.input_file}}")
    else:
        km.execute(*(parthenon_args or []))
'''
        )
        exec(
            run_source,
            {**exec_globals, "self": self, "params_dict": params_dict},
            exec_locals,
        )

        generate_input_source = (
            """
@self._app.command("generate-input")
def generate_input(
            """
            + param_args
            + '''
            output: Optional[str] = typer.Option(
                None, "-o", "--output", help="Output file path (default: {name}.in)"
            ),
        ):
    """Generate the Parthenon input file."""
    sim_kwargs = {{k: v for k, v in locals().items() if k in params_dict}}
    km = self.func(**sim_kwargs)
    if output:
        km.write_input(file=output)
        typer.echo(f"Input file written to: {{output}}")
    else:
        km.write_input()
        typer.echo(f"Input file written to: {{km.input_file}}")
'''.format(name=self.name)
        )
        exec(
            generate_input_source,
            {**exec_globals, "self": self, "params_dict": params_dict},
            exec_locals,
        )

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

    def __call__(self, *args: Any, **kwargs: Any) -> "KamayanManager":
        """Call the wrapped simulation function.

        Args:
            *args: Positional arguments to pass to the simulation function.
            **kwargs: Keyword arguments to pass to the simulation function.

        Returns:
            The KamayanManager instance from the simulation function.
        """
        return self.func(*args, **kwargs)

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
        def wrapper(*args: Any, **kwargs: Any) -> "KamayanManager":
            return func(*args, **kwargs)

        cast(Any, wrapper).app = sim.app
        return cast("KamayanSimulation", wrapper)

    return decorator
