"""Atlas-RAG CLI Application (Typer-based)."""
import typer
from typing_extensions import Annotated

from src.core.cli.commands.chunk import chunk_command
from src.core.cli.commands.batch import batch_command
from src.core.cli.commands.search import search_command
from src.core.cli.commands.info import info_command
from src.core.cli.commands.ingest import ingest_command
from src.core.cli.commands.eval import eval_command
from src.core.cli.commands.retry import retry_command


# Create main Typer app
app = typer.Typer(
    name="atlas-rag",
    help="Atlas-RAG CLI - Production-ready document processing for RAG applications",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    """Display version and exit."""
    if value:
        from importlib.metadata import version, PackageNotFoundError
        try:
            app_version = version("atlas-rag")
        except PackageNotFoundError:
            app_version = "0.1.0 (dev)"

        typer.echo(f"Atlas-RAG CLI version {app_version}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True
        )
    ] = False,
):
    """
    Atlas-RAG CLI - Document processing and semantic search.

    A production-ready pipeline for chunking documents and building
    RAG (Retrieval-Augmented Generation) applications.

    \b
    Quick Start:
        1. Chunk a document:
           $ atlas-rag chunk document.txt --show

        2. Process multiple files:
           $ atlas-rag batch ./documents -o chunks.json

        3. Ingest to vector store:
           $ atlas-rag ingest chunks.json

        4. Evaluate chunking quality:
           $ atlas-rag eval chunks.json

        5. System info:
           $ atlas-rag info

    \b
    Documentation:
        https://github.com/horiz-data/atlas-rag

    \b
    Support:
        Report issues at: https://github.com/horiz-data/atlas-rag/issues
    """
    pass


# Register commands
app.command(name="chunk", help="Chunk a single document")(chunk_command)
app.command(name="batch", help="Process multiple files in batch mode")(batch_command)
app.command(name="ingest", help="Ingest chunks into Qdrant vector store")(ingest_command)
app.command(name="search", help="Search in vector store using semantic search", hidden=True)(search_command)
app.command(name="eval", help="Evaluate chunking quality and compare strategies")(eval_command)
app.command(name="info", help="Display system information and status")(info_command)
app.command(name="retry", help="Retry failed files from a previous run")(retry_command)


if __name__ == "__main__":
    app()
