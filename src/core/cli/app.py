"""Atlas-RAG CLI Application (Typer-based)."""
import warnings
import typer
from typing_extensions import Annotated

# Suppress common warnings for better UX
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")

# Create main Typer app
app = typer.Typer(
    name="ragctl",
    help="RAG Studio - Production-ready RAG toolkit with intelligent document processing",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    """Display version and exit."""
    if value:
        from importlib.metadata import version, PackageNotFoundError
        try:
            app_version = version("ragctl")
        except PackageNotFoundError:
            try:
                # Fallback to old name for development
                app_version = version("atlas-rag")
            except PackageNotFoundError:
                app_version = "0.1.0 (dev)"

        typer.echo(f"RAG Studio (ragctl) version {app_version}")
        raise typer.Exit()


# Lazy import - commands are imported only when actually used
# This makes --version and --help instant instead of loading all heavy dependencies


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
    RAG Studio - Production-ready RAG toolkit with intelligent document processing.

    A comprehensive CLI for building RAG (Retrieval-Augmented Generation) systems
    with advanced OCR, semantic chunking, and vector store integration.

    \b
    Quick Start:
        1. Chunk a document:
           $ ragctl chunk document.txt --show

        2. Process multiple files:
           $ ragctl batch ./documents -o chunks.json

        3. Ingest to vector store:
           $ ragctl ingest chunks.json

        4. Evaluate chunking quality:
           $ ragctl eval chunks.json

        5. System info:
           $ ragctl info

    \b
    Documentation:
        https://github.com/horiz-data/atlas-rag

    \b
    Support:
        Report issues at: https://github.com/horiz-data/atlas-rag/issues
    """
    pass


# Register commands - imports happen inside each command function (lazy loading)
@app.command(name="chunk", help="Chunk a single document")
def chunk(
    input_file: str = typer.Argument(..., help="Input file to chunk"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    strategy: str = typer.Option("semantic", "--strategy", "-s", help="Chunking strategy"),
    max_tokens: int = typer.Option(400, "--max-tokens", help="Maximum tokens per chunk"),
    overlap: int = typer.Option(50, "--overlap", help="Overlap between chunks"),
    show: bool = typer.Option(False, "--show", help="Display chunks in terminal"),
    advanced_ocr: bool = typer.Option(False, "--advanced-ocr", help="Use advanced OCR"),
):
    """Chunk a single document."""
    from src.core.cli.commands.chunk import chunk_command
    return chunk_command(input_file, output, strategy, max_tokens, overlap, show, advanced_ocr)


@app.command(name="batch", help="Process multiple files in batch mode")
def batch(
    input_dir: str = typer.Argument(..., help="Input directory with files"),
    output: str = typer.Option(None, "--output", "-o", help="Output directory or file"),
    pattern: str = typer.Option("*", "--pattern", "-p", help="File pattern to match"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process recursively"),
    auto_continue: bool = typer.Option(False, "--auto-continue", help="Continue on errors"),
):
    """Process multiple files in batch mode."""
    from src.core.cli.commands.batch import batch_command
    return batch_command(input_dir, output, pattern, recursive, auto_continue)


@app.command(name="ingest", help="Ingest chunks into Qdrant vector store")
def ingest(
    input_file: str = typer.Argument(..., help="Input JSON/JSONL file with chunks"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
    url: str = typer.Option("http://localhost:6333", "--url", help="Qdrant URL"),
):
    """Ingest chunks into Qdrant vector store."""
    from src.core.cli.commands.ingest import ingest_command
    return ingest_command(input_file, collection, url)


@app.command(name="search", help="Search in vector store using semantic search", hidden=True)
def search(
    query: str = typer.Argument(..., help="Search query"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection name"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results"),
):
    """Search in vector store using semantic search."""
    from src.core.cli.commands.search import search_command
    return search_command(query, collection, limit)


@app.command(name="eval", help="Evaluate chunking quality and compare strategies")
def eval(
    input_file: str = typer.Argument(..., help="Input file to evaluate"),
    strategies: str = typer.Option("semantic,sentence,token", "--strategies", help="Strategies to compare"),
):
    """Evaluate chunking quality and compare strategies."""
    from src.core.cli.commands.eval import eval_command
    return eval_command(input_file, strategies)


@app.command(name="info", help="Display system information and status")
def info(
    api_url: str = typer.Option("http://localhost:8000", "--api-url", help="API server URL"),
):
    """Display system information and status."""
    from src.core.cli.commands.info import info_command
    return info_command(api_url)


@app.command(name="retry", help="Retry failed files from a previous run")
def retry(
    run_id: str = typer.Argument(None, help="Run ID to retry (optional)"),
    show: bool = typer.Option(False, "--show", help="Show failed runs"),
):
    """Retry failed files from a previous run."""
    from src.core.cli.commands.retry import retry_command
    return retry_command(run_id, show)


if __name__ == "__main__":
    app()
