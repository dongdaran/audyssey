from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import typer
import uvicorn

from audyssey.api import create_app
from audyssey.settings import Settings
from audyssey.toolserver import create_tool_app


class Tool(str, Enum):
    separation = "separation"
    filtering = "filtering"
    grounding = "grounding"
    generation = "generation"


app = typer.Typer()


@app.callback()
def callback() -> None:
    """Audyssey audio editing skeleton."""


@app.command()
def batch(
    videos: Annotated[
        list[Path], typer.Argument(help="Input videos, processed in order.")
    ],
    mode: Annotated[
        Literal["separate", "generate"], typer.Option(help="Required processing mode.")
    ],
    render: Annotated[bool, typer.Option(help="Export mix, video and stems.")] = True,
) -> None:
    """Process videos locally and append batch.csv and timings.csv."""
    from audyssey.batch import run_batch

    if run_batch(videos, Settings(), render, mode):
        raise typer.Exit(1)


@app.command()
def results(
    root: Annotated[Path, typer.Argument(help="Batch results root directory.")] = Path("/audyssey-results"),
    host: str = "0.0.0.0",
    port: int = 8610,
) -> None:
    """Browse saved batch results without loading models."""
    from audyssey.results import create_results_app
    typer.echo(f"Results: http://<server-ip>:{port} (root: {root})")
    uvicorn.run(create_results_app(root), host=host, port=port)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8600) -> None:
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def tool_server(
    tool: Annotated[list[Tool], typer.Option()],
    impl: Annotated[Literal["stub", "real"], typer.Option()] = "stub",
    host: str = "127.0.0.1",
    port: int = 8700,
) -> None:
    token = Settings().tool_token
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        typer.echo(
            "AUDYSSEY_TOOL_TOKEN is required when binding to a non-loopback host"
        )
        raise typer.Exit(1)
    uvicorn.run(
        create_tool_app([item.value for item in tool], impl, token),
        host=host,
        port=port,
    )


def main() -> None:
    app()
