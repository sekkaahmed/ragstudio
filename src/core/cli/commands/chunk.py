"""Chunk command for Atlas-RAG CLI."""
import json
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any
from typing_extensions import Annotated
from datetime import datetime
import time

import typer

from src.core.chunk.chunker import chunk_document
from src.core.cli.utils.display import (
    console,
    print_success,
    print_error,
    create_chunks_table,
    display_stats
)
from src.core.cli.utils.validation import validate_file_exists, validate_output_path
from src.core.cli.utils.output import save_chunks, detect_format_from_extension, OutputFormat
from src.core.cli.utils.security import (
    validate_file_size,
    validate_no_symlinks,
    validate_disk_space,
    sanitize_metadata,
    get_security_config
)


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""
    semantic = "semantic"
    sentence = "sentence"
    token = "token"  # nosec B105 - Strategy name, not a password


def _display_routing_decisions(result: Dict[str, Any]) -> None:
    """
    Display OCR routing decisions in a clear, structured format.

    Args:
        result: OCR processing result with routing_decisions list
    """
    if not result.get("routing_decisions"):
        return

    console.print("\n[bold cyan]📋 OCR Routing Decision Tree:[/bold cyan]")

    decisions = result["routing_decisions"]
    for i, decision in enumerate(decisions, 1):
        step = decision.get("step", "unknown")

        if step == "ocr_quality_detection":
            quality_category = decision.get("ocr_quality_category", "UNKNOWN")
            quality_score = decision.get("ocr_quality_score", 0.0)
            recommended_engine = decision.get("recommended_engine", "unknown")

            # Color based on quality
            if quality_category == "HIGH":
                quality_color = "green"
            elif quality_category == "MEDIUM":
                quality_color = "yellow"
            else:
                quality_color = "red"

            console.print(f"  [{i}] [bold]OCR Quality Detection[/bold]")
            console.print(f"      • Quality: [{quality_color}]{quality_category}[/{quality_color}] (score: {quality_score:.3f})")
            console.print(f"      • Recommended: [bold]{recommended_engine}[/bold]")
            console.print(f"      → [italic]Reason:[/italic] {'Low quality requires advanced OCR' if quality_category == 'LOW' else 'Standard OCR sufficient'}")

        elif step == "scientific_detection":
            is_scientific = decision.get("is_scientific", False)
            math_density = decision.get("math_density", 0.0)
            recommended_engine = decision.get("recommended_engine", "unknown")

            console.print(f"  [{i}] [bold]Scientific Content Detection[/bold]")
            if is_scientific:
                console.print(f"      • Scientific: [bold green]YES[/bold green] (math density: {math_density:.3f})")
                console.print(f"      • Recommended: [bold]{recommended_engine}[/bold]")
                console.print(f"      → [italic]Reason:[/italic] High mathematical content requires specialized OCR (Nougat)")
            else:
                console.print(f"      • Scientific: [dim]NO[/dim] (math density: {math_density:.3f})")
                console.print(f"      → [italic]Reason:[/italic] No specialized mathematical OCR needed")

        elif step == "complexity_analysis":
            complexity_score = decision.get("complexity_score", 0.0)
            recommended_strategy = decision.get("recommended_strategy", "unknown")

            # Determine complexity level
            if complexity_score >= 0.7:
                complexity_level = "HIGH"
                complexity_color = "red"
                reason = "Complex document requires advanced OCR (Qwen-VL)"
            elif complexity_score >= 0.4:
                complexity_level = "MEDIUM"
                complexity_color = "yellow"
                reason = "Moderate complexity - standard or mid-tier OCR suitable"
            else:
                complexity_level = "LOW"
                complexity_color = "green"
                reason = "Simple document - classic OCR sufficient"

            console.print(f"  [{i}] [bold]Complexity Analysis[/bold]")
            console.print(f"      • Complexity: [{complexity_color}]{complexity_level}[/{complexity_color}] (score: {complexity_score:.3f})")
            console.print(f"      • Recommended: [bold]{recommended_strategy}[/bold]")
            console.print(f"      → [italic]Reason:[/italic] {reason}")

        elif step == "ocr_routing":
            engine_used = decision.get("engine_used", "unknown")
            routing_reason = decision.get("routing_reason", "")

            # Determine engine type for color
            if "qwen" in engine_used.lower():
                engine_color = "magenta"
                engine_type = "Advanced Vision-Language Model"
            elif "nougat" in engine_used.lower():
                engine_color = "blue"
                engine_type = "Scientific OCR Specialist"
            elif "classic" in engine_used.lower():
                engine_color = "cyan"
                engine_type = "Standard OCR"
            else:
                engine_color = "white"
                engine_type = "OCR Engine"

            console.print(f"  [{i}] [bold]Final OCR Engine Selection[/bold]")
            console.print(f"      • Engine: [{engine_color}]{engine_used}[/{engine_color}]")
            console.print(f"      • Type: [dim]{engine_type}[/dim]")
            console.print(f"      → [italic]Reason:[/italic] {routing_reason}")

            # Check for fallback indication
            if "fallback" in routing_reason.lower():
                console.print(f"      [yellow]⚠ Note:[/yellow] Primary engine unavailable, using fallback")

        elif step == "fallback":
            engine_used = decision.get("engine_used", "unknown")
            reason = decision.get("reason", "")

            console.print(f"  [{i}] [bold yellow]Fallback Activated[/bold yellow]")
            console.print(f"      • Fallback engine: [bold]{engine_used}[/bold]")
            console.print(f"      → [italic]Reason:[/italic] {reason}")

    console.print("")


def _generate_processing_summary(
    file_path: Path,
    config: Any,
    processing_data: Dict[str, Any],
    chunks: list,
    success: bool = True,
    errors: list = None
) -> Dict[str, Any]:
    """
    Generate structured JSON summary of the processing pipeline.

    Args:
        file_path: Path to the processed file
        config: AtlasConfig instance
        processing_data: Dictionary with processing information (OCR results, timings, etc.)
        chunks: List of generated chunks
        success: Whether processing was successful
        errors: List of error messages if any

    Returns:
        Dictionary with complete processing summary
    """
    summary = {
        "metadata": {
            "atlas_rag_version": "1.0.0",
            "processing_timestamp": datetime.now().isoformat(),
            "success": success,
            "errors": errors or []
        },
        "document": {
            "path": str(file_path),
            "filename": file_path.name,
            "format": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "text_length": processing_data.get("text_length", 0),
            "language": processing_data.get("language", "unknown")
        },
        "configuration": {
            "llm": {
                "enabled": config.llm.use_llm,
                "provider": config.llm.provider if config.llm.use_llm else None,
                "model": config.llm.model if config.llm.use_llm else None,
                "is_local": config.llm.is_local if config.llm.use_llm else None
            },
            "ocr": {
                "advanced_ocr_enabled": config.ocr.use_advanced_ocr,
                "dictionary_threshold": config.ocr.dictionary_threshold,
                "dynamic_threshold": config.ocr.dynamic_threshold,
                "fallback_enabled": config.ocr.enable_fallback
            },
            "chunking": {
                "strategy": config.chunking.strategy,
                "max_tokens": config.chunking.max_tokens,
                "overlap": config.chunking.overlap
            }
        },
        "processing": {
            "total_time_seconds": processing_data.get("total_time", 0),
            "stages": {}
        },
        "results": {
            "chunks": {
                "total_count": len(chunks),
                "average_size_chars": (sum(len(c.text) for c in chunks) // len(chunks)) if len(chunks) > 0 else 0,
                "min_size_chars": min((len(c.text) for c in chunks), default=0),
                "max_size_chars": max((len(c.text) for c in chunks), default=0),
                "total_text_length": sum(len(c.text) for c in chunks)
            }
        }
    }

    # Add OCR-specific data if available
    if processing_data.get("ocr_result"):
        ocr_data = processing_data["ocr_result"]
        summary["processing"]["stages"]["ocr"] = {
            "time_seconds": processing_data.get("ocr_time", 0),
            "engine": ocr_data.get("metadata", {}).get("ocr_engine", "unknown"),
            "success": ocr_data.get("metadata", {}).get("success", False),
            "routing_decisions": ocr_data.get("routing_decisions", []),
            "quality_metrics": ocr_data.get("metadata", {}).get("quality_metrics", {}),
            "fallback_used": ocr_data.get("metadata", {}).get("fallback_from") is not None,
            "fallback_reason": ocr_data.get("metadata", {}).get("fallback_reason")
        }

    # Add strategy selection data if available
    if processing_data.get("strategy_selection"):
        summary["processing"]["stages"]["strategy_selection"] = processing_data["strategy_selection"]

    # Add chunking timing
    if processing_data.get("chunking_time"):
        summary["processing"]["stages"]["chunking"] = {
            "time_seconds": processing_data["chunking_time"]
        }

    return summary


# Simple Document class for CLI
class Document:
    """Simple document wrapper for chunking."""
    def __init__(self, text: str, metadata: dict, source_path: str, id: Optional[str] = None):
        self.text = text
        self.metadata = metadata
        self.source_path = source_path
        self.id = id


def _load_document_universal(file_path: Path, print_info_func, use_status: bool = True) -> str:
    """
    Universal document loader supporting all formats.

    Supports: TXT, MD, PDF, DOCX, DOC, HTML, HTM, PNG, JPG, JPEG, TIFF

    Args:
        file_path: Path to the file
        print_info_func: Function to print info messages
        use_status: Whether to use console.status (disable in batch mode to avoid conflicts)

    Returns:
        Extracted text content
    """
    suffix = file_path.suffix.lower()

    # Fast path for simple text files
    if suffix in {'.txt', '.md'}:
        if use_status:
            with console.status(f"[bold green]Reading {file_path.name}..."):
                return file_path.read_text(encoding='utf-8')
        else:
            return file_path.read_text(encoding='utf-8')

    # Use universal loader for all other formats (PDF, Word, HTML, images, etc.)
    try:
        from src.workflows.ingest.loader import ingest_file, SUPPORTED_EXTENSIONS

        # Check if format is supported
        if suffix not in SUPPORTED_EXTENSIONS:
            print_info_func(f"⚠️  Format {suffix} not officially supported, attempting to load...")

        if use_status:
            with console.status(f"[bold green]Loading {file_path.name} ({suffix})..."):
                document = ingest_file(str(file_path))
                text = document.text
        else:
            document = ingest_file(str(file_path))
            text = document.text

        if not text:
            raise ValueError(f"No text extracted from {file_path.name}")

        # Show info about extraction (only if not in batch mode)
        if use_status:
            print_info_func(f"Extracted [bold]{len(text)} chars[/bold] from {suffix.upper()} document")

        return text

    except ImportError as e:
        # Fallback to simple text reading if loader not available
        from src.core.cli.utils.display import print_warning
        if use_status:
            print_warning(f"Universal loader not available ({e}), trying simple text read")
        return file_path.read_text(encoding='utf-8')

    except Exception as e:
        raise RuntimeError(f"Failed to load document: {e}")


def chunk_command(
    file: Annotated[
        Path,
        typer.Argument(
            help="File to chunk",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        )
    ],
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy", "-s",
            help="Chunking strategy to use (semantic, sentence, token, or auto for ML detection)"
        )
    ] = "semantic",
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens", "-m",
            help="Maximum tokens per chunk",
            min=50,
            max=2000
        )
    ] = 400,
    overlap: Annotated[
        int,
        typer.Option(
            "--overlap", "-ol",
            help="Token overlap between chunks",
            min=0,
            max=500
        )
    ] = 50,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output", "-o",
            help="Output JSON file path"
        )
    ] = None,
    show: Annotated[
        bool,
        typer.Option(
            "--show",
            help="Display chunks table in terminal"
        )
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-l",
            help="Maximum number of chunks to display",
            min=1,
            max=100
        )
    ] = 10,
    advanced_ocr: Annotated[
        bool,
        typer.Option(
            "--advanced-ocr",
            help="Use intelligent OCR routing for scanned PDFs (auto-selects Qwen-VL, Nougat, or classic OCR)"
        )
    ] = False,
    # LLM Configuration
    use_llm: Annotated[
        bool,
        typer.Option(
            "--use-llm",
            help="Enable LLM for text correction and analysis"
        )
    ] = False,
    llm_url: Annotated[
        Optional[str],
        typer.Option(
            "--llm-url",
            help="LLM API URL (auto-detected for local Ollama if not specified)"
        )
    ] = None,
    llm_model: Annotated[
        Optional[str],
        typer.Option(
            "--llm-model",
            help="LLM model name (e.g., mistral:latest, gpt-4o-mini)"
        )
    ] = None,
    llm_provider: Annotated[
        Optional[str],
        typer.Option(
            "--llm-provider",
            help="LLM provider (ollama, openai, anthropic)"
        )
    ] = None,
    # OCR Configuration
    ocr_threshold: Annotated[
        Optional[float],
        typer.Option(
            "--ocr-threshold",
            help="Dictionary ratio threshold for OCR quality (0.0-1.0, default: 0.30)",
            min=0.0,
            max=1.0
        )
    ] = None,
    ocr_dynamic_threshold: Annotated[
        Optional[bool],
        typer.Option(
            "--ocr-dynamic-threshold/--no-ocr-dynamic-threshold",
            help="Enable dynamic threshold adjustment based on language and text length"
        )
    ] = None,
    ocr_fallback: Annotated[
        Optional[bool],
        typer.Option(
            "--ocr-fallback/--no-ocr-fallback",
            help="Enable fallback to classic OCR on advanced OCR failures"
        )
    ] = None,
    # Configuration file
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config", "-c",
            help="Path to configuration file (YAML)"
        )
    ] = None,
    # Output options
    generate_summary: Annotated[
        bool,
        typer.Option(
            "--summary",
            help="Generate processing summary JSON"
        )
    ] = False,
) -> None:
    """
    Chunk a document using the specified strategy.

    This command reads a text file and splits it into semantically meaningful
    chunks optimized for RAG applications.

    \b
    Examples:
        # Basic chunking with semantic strategy
        ragctl chunk document.txt

        # Chunk with specific parameters
        ragctl chunk document.txt --strategy token --max-tokens 512

        # Chunk and save to JSON
        ragctl chunk document.txt -o chunks.json

        # Chunk and display results
        ragctl chunk document.txt --show --limit 5
    """
    # === SECURITY VALIDATIONS ===
    security_config = get_security_config()

    # Validate file is not a symlink
    validate_no_symlinks(file, security_config)

    # Validate file size
    validate_file_size(file, security_config, warn_only=False)

    # Validate output disk space if output specified
    if output:
        validate_disk_space(output, security_config)

    # === CONFIGURATION LOADING (Hierarchy: CLI > ENV > YAML > Defaults) ===
    from src.core.config.atlas_config import AtlasConfig, get_atlas_config

    # Load base config (from file if specified, or from default ~/.atlasrag/config.yml + ENV)
    if config_file:
        try:
            config = AtlasConfig.from_file(config_file)
            console.print(f"[bold cyan]ℹ[/bold cyan] Loaded config from: [bold]{config_file}[/bold]")
        except FileNotFoundError:
            print_error(f"Config file not found: {config_file}")
            raise typer.Exit(code=1)
        except Exception as e:
            print_error(f"Error loading config file: {e}")
            raise typer.Exit(code=1)
    else:
        # Load from default locations (~/.atlasrag/config.yml + ENV)
        config = get_atlas_config()

    # Override with CLI arguments (highest priority)
    cli_overrides = {}

    # LLM overrides
    if use_llm:
        cli_overrides['use_llm'] = use_llm
    if llm_url:
        cli_overrides['llm_url'] = llm_url
    if llm_model:
        cli_overrides['llm_model'] = llm_model
    if llm_provider:
        cli_overrides['llm_provider'] = llm_provider

    # OCR overrides
    if advanced_ocr:
        cli_overrides['use_advanced_ocr'] = advanced_ocr
        cli_overrides['advanced_ocr'] = advanced_ocr  # For backward compatibility
    if ocr_threshold is not None:
        cli_overrides['ocr_threshold'] = ocr_threshold
    if ocr_dynamic_threshold is not None:
        cli_overrides['ocr_dynamic_threshold'] = ocr_dynamic_threshold
    if ocr_fallback is not None:
        cli_overrides['ocr_fallback'] = ocr_fallback

    # Chunking overrides
    cli_overrides['strategy'] = strategy
    cli_overrides['max_tokens'] = max_tokens
    cli_overrides['overlap'] = overlap

    # Output overrides
    if generate_summary:
        cli_overrides['generate_summary'] = generate_summary

    # Apply CLI overrides
    if cli_overrides:
        config.merge_from_cli_args(**cli_overrides)

    # Validate configuration
    is_valid, error = config.llm.validate()
    if not is_valid:
        print_error(f"Invalid LLM configuration: {error}")
        raise typer.Exit(code=1)

    # Log effective configuration (INFO level)
    from src.core.cli.utils.display import print_info
    if config.llm.use_llm:
        llm_type = "local" if config.llm.is_local else "remote"
        print_info(f"Using {llm_type} LLM: [bold]{config.llm.url}[/bold] (model: {config.llm.model})")

    if config.ocr.use_advanced_ocr:
        threshold_info = f"dynamic ({config.ocr.dictionary_threshold:.2f} base)" if config.ocr.dynamic_threshold else f"fixed ({config.ocr.dictionary_threshold:.2f})"
        print_info(f"Advanced OCR enabled with {threshold_info} threshold")

    # === PROCESSING DATA COLLECTION (for summary) ===
    start_time = time.time()
    processing_data: Dict[str, Any] = {
        "text_length": 0,
        "language": "unknown",
        "ocr_result": None,
        "ocr_time": 0,
        "strategy_selection": None,
        "chunking_time": 0,
        "total_time": 0
    }
    errors_list = []

    # === VALIDATION ===

    # Validate file
    try:
        validate_file_exists(file)
    except typer.BadParameter as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    # Validate output path
    if output:
        try:
            validate_output_path(output)
        except typer.BadParameter as e:
            print_error(str(e))
            raise typer.Exit(code=1)

    # Read file (with support for all document formats)
    try:
        # Define document formats that support advanced OCR routing
        ADVANCED_OCR_FORMATS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff'}

        if config.ocr.use_advanced_ocr and file.suffix.lower() in ADVANCED_OCR_FORMATS:
            # Use intelligent OCR router for PDFs and images
            from src.core.cli.utils.display import print_info, print_warning

            console.print(f"\n[bold cyan]🔍 Using intelligent OCR routing for {file.suffix.upper()}...[/bold cyan]")

            try:
                # Lazy import to avoid loading heavy dependencies unless needed
                from src.workflows.router.ocr_router import OCRRouter, OCRRouterConfig

                ocr_start_time = time.time()

                with console.status("[bold cyan]Initializing OCR engines..."):
                    # Create OCRRouterConfig from AtlasConfig
                    router_config = OCRRouterConfig()
                    # TODO: Pass atlas config values to router_config
                    router = OCRRouter(router_config)

                with console.status(f"[bold green]Processing {file.suffix.upper()} with intelligent OCR..."):
                    result = router.process_document(file)

                ocr_end_time = time.time()
                processing_data["ocr_time"] = ocr_end_time - ocr_start_time
                processing_data["ocr_result"] = result

                text = result['extracted_text']

                # Display OCR info
                ocr_engine = result['metadata'].get('ocr_engine', 'unknown')
                success = result['metadata'].get('success', False)

                if success:
                    print_info(f"OCR Engine: [bold]{ocr_engine}[/bold]")
                    print_info(f"Processing time: [bold]{result['processing_time']:.2f}s[/bold]")
                    print_info(f"Extracted text: [bold]{len(text)} chars[/bold]")

                    # Display structured routing decision tree
                    _display_routing_decisions(result)
                else:
                    print_warning("OCR extraction had issues, check errors below")
                    if result.get('errors'):
                        for error in result['errors']:
                            print_error(f"  • {error}")

                if not text:
                    print_error("No text extracted from PDF")
                    raise typer.Exit(code=1)

            except ImportError as e:
                print_warning(f"Advanced OCR dependencies not available: {e}")
                print_info("Falling back to standard document loading")
                # Fallback to universal loader
                from src.core.cli.utils.display import print_info as pinfo
                text = _load_document_universal(file, pinfo)
            except Exception as e:
                print_error(f"Advanced OCR failed: {e}")
                print_info("Falling back to standard document loading")
                # Fallback to universal loader
                from src.core.cli.utils.display import print_info as pinfo
                text = _load_document_universal(file, pinfo)
        else:
            # Universal document loading for all formats
            from src.core.cli.utils.display import print_info
            text = _load_document_universal(file, print_info)
    except Exception as e:
        print_error(f"Error reading file: {e}")
        raise typer.Exit(code=1)

    # Create document
    # Create and sanitize metadata
    raw_metadata = {"source": str(file), "size": len(text)}
    sanitized_metadata = sanitize_metadata(raw_metadata, security_config)

    doc = Document(
        text=text,
        metadata=sanitized_metadata,
        source_path=str(file)
    )

    # Update processing data with text length
    processing_data["text_length"] = len(text)

    # Handle empty files
    if not text or len(text.strip()) == 0:
        from src.core.cli.utils.display import print_warning
        print_warning(f"Empty file: {file.name} contains no text")
        return 0

    # Handle auto strategy selection
    if config.chunking.strategy == "auto":
        console.print(f"\n[bold cyan]🤖 Analyzing document for optimal strategy...[/bold cyan]")

        try:
            from src.core.cli.utils.display import print_info
            # Lazy import to avoid NumPy issues
            from src.workflows.analyzer.strategy_selector import predict_best_strategy

            strategy_start_time = time.time()
            with console.status("[bold cyan]Running ML prediction..."):
                predicted_strategy, confidence = predict_best_strategy(text)
            strategy_end_time = time.time()

            print_info(
                f"Recommended strategy: [bold]{predicted_strategy}[/bold] "
                f"(confidence: {confidence:.1%})"
            )

            # Capture strategy selection data
            processing_data["strategy_selection"] = {
                "method": "ml_prediction",
                "original_strategy": "auto",
                "selected_strategy": predicted_strategy,
                "confidence": confidence,
                "time_seconds": strategy_end_time - strategy_start_time
            }

            config.chunking.strategy = predicted_strategy

        except ImportError as e:
            from src.core.cli.utils.display import print_warning
            print_warning(f"ML prediction unavailable (dependency issue)")
            print_info("Falling back to semantic strategy")
            config.chunking.strategy = "semantic"

        except Exception as e:
            from src.core.cli.utils.display import print_warning
            print_warning(f"ML prediction failed: {e}")
            print_info("Falling back to semantic strategy")
            config.chunking.strategy = "semantic"

    # Validate strategy
    valid_strategies = ["semantic", "sentence", "token"]
    if config.chunking.strategy not in valid_strategies:
        print_error(f"Invalid strategy: {config.chunking.strategy}. Must be one of: {', '.join(valid_strategies)}, auto")
        raise typer.Exit(code=1)

    # Chunk document (using config values)
    chunking_start_time = time.time()
    with console.status(f"[bold green]Chunking {file.name} with {config.chunking.strategy} strategy..."):
        try:
            chunks = chunk_document(
                doc,
                strategy=config.chunking.strategy,
                max_tokens=config.chunking.max_tokens,
                overlap=config.chunking.overlap
            )
        except Exception as e:
            print_error(f"Chunking error: {e}")
            errors_list.append(f"Chunking error: {e}")
            raise typer.Exit(code=1)
    chunking_end_time = time.time()
    processing_data["chunking_time"] = chunking_end_time - chunking_start_time

    # Display results
    print_success(f"Successfully chunked [bold]{file.name}[/bold]")

    stats = {
        "Strategy": config.chunking.strategy,
        "Chunks created": len(chunks),
        "Average chunk size": f"{sum(len(c.text) for c in chunks) // len(chunks)} chars",
        "Max tokens": config.chunking.max_tokens,
        "Overlap": config.chunking.overlap
    }
    display_stats(stats)

    # Show chunks table
    if show:
        chunks_data = [{"id": c.id, "text": c.text} for c in chunks]
        table = create_chunks_table(chunks_data, title=f"Chunks from {file.name}", limit=limit)
        console.print("\n")
        console.print(table)

    # Save to file if requested
    if output:
        chunks_data = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

        try:
            # Auto-detect format from extension
            output_format = detect_format_from_extension(output)

            with console.status(f"[bold green]Saving to {output.name} ({output_format.value})..."):
                save_chunks(chunks_data, output, output_format)

            print_success(f"Chunks saved to [bold]{output}[/bold] ({output_format.value} format)")
        except Exception as e:
            print_error(f"Error saving output: {e}")
            errors_list.append(f"Error saving output: {e}")
            raise typer.Exit(code=1)

    # Generate processing summary if requested
    if config.output.generate_summary:
        end_time = time.time()
        processing_data["total_time"] = end_time - start_time

        # Generate summary
        summary = _generate_processing_summary(
            file_path=file,
            config=config,
            processing_data=processing_data,
            chunks=chunks,
            success=len(errors_list) == 0,
            errors=errors_list
        )

        # Determine summary output path
        if output:
            # Save summary next to output file
            summary_path = output.parent / f"{output.stem}_summary.json"
        else:
            # Save summary next to source file
            summary_path = file.parent / f"{file.stem}_processing_summary.json"

        try:
            with console.status(f"[bold green]Generating processing summary..."):
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)

            print_success(f"Processing summary saved to [bold]{summary_path}[/bold]")

            # Display key metrics from summary
            console.print("\n[bold cyan]📊 Processing Summary:[/bold cyan]")
            console.print(f"  • Total time: [bold]{summary['processing']['total_time_seconds']:.2f}s[/bold]")
            console.print(f"  • Document size: [bold]{summary['document']['text_length']:,} chars[/bold]")
            console.print(f"  • Chunks generated: [bold]{summary['results']['chunks']['total_count']}[/bold]")

            if summary['processing']['stages'].get('ocr'):
                ocr_stage = summary['processing']['stages']['ocr']
                console.print(f"  • OCR engine: [bold]{ocr_stage['engine']}[/bold] ({ocr_stage['time_seconds']:.2f}s)")
                if ocr_stage.get('fallback_used'):
                    console.print(f"    [yellow]⚠ Fallback used: {ocr_stage['fallback_reason']}[/yellow]")

            if summary['processing']['stages'].get('strategy_selection'):
                strategy = summary['processing']['stages']['strategy_selection']
                console.print(f"  • Strategy: [bold]{strategy['selected_strategy']}[/bold] (confidence: {strategy['confidence']:.1%})")

        except Exception as e:
            print_error(f"Error saving summary: {e}")
            errors_list.append(f"Error saving summary: {e}")
            # Don't raise - summary generation is optional
