"""Test that all dependencies are installed correctly."""

try:
    import langgraph
    import langchain
    import chromadb
    import sentence_transformers
    from rich.console import Console
    from dotenv import load_dotenv
    import pydantic
    import numpy

    console = Console()
    console.print("[green]✓ All dependencies installed successfully![/green]")
    # console.print(f"[cyan]LangGraph version: {langgraph.__version__}[/cyan]")  # Not available in all versions
    console.print(f"[cyan]LangChain version: {langchain.__version__}[/cyan]")
    console.print(f"[cyan]ChromaDB version: {chromadb.__version__}[/cyan]")

except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
