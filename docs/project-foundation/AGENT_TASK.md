# Agent Task: Project Foundation

## Branch: `feature/project-foundation`

## Priority: CRITICAL - Must complete first (BLOCKING)

## Objective
Setup the complete project foundation using `uv` as the package manager, including directory structure, dependencies, and configuration templates.

## Tasks

### 1. Initialize uv Project
```bash
cd worktrees/project-foundation
uv init --name hvas-mini --app
```

### 2. Create `pyproject.toml`
Configure with all required dependencies as specified in spec.md:

```toml
[project]
name = "hvas-mini"
version = "0.1.0"
description = "HVAS Mini Prototype - Hierarchical Vector Agent System with LangGraph"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-anthropic>=0.2.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "mypy>=1.8.0",
    "ruff>=0.2.0",
]
```

### 3. Create Directory Structure
```bash
mkdir -p src/hvas_mini
mkdir -p data/memories
mkdir -p logs/runs
mkdir -p docs
touch src/hvas_mini/__init__.py
```

### 4. Create `.env.example`
Full configuration template (DO NOT include actual API keys):

```bash
# LLM Configuration
ANTHROPIC_API_KEY=your_key_here
MODEL_NAME=claude-3-haiku-20240307
BASE_TEMPERATURE=0.7

# Memory Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
MEMORY_SCORE_THRESHOLD=7.0
MAX_MEMORIES_RETRIEVE=3

# Evolution Configuration
ENABLE_PARAMETER_EVOLUTION=true
EVOLUTION_LEARNING_RATE=0.1
MIN_TEMPERATURE=0.5
MAX_TEMPERATURE=1.0

# Visualization
ENABLE_VISUALIZATION=true
SHOW_MEMORY_RETRIEVAL=true
SHOW_PARAMETER_CHANGES=true

# LangGraph Configuration
STREAM_MODE=values
RECURSION_LIMIT=10
```

### 5. Create `src/hvas_mini/__init__.py`
```python
"""
HVAS Mini - Hierarchical Vector Agent System Prototype

A demonstration of concurrent AI agents with individual RAG memory,
parameter evolution, and real-time learning visualization using LangGraph.
"""

__version__ = "0.1.0"
```

### 6. Install Dependencies
```bash
uv sync
```

### 7. Verify Installation
Create a simple test script to verify imports work:

```python
# test_imports.py
try:
    import langgraph
    import langchain
    import chromadb
    import sentence_transformers
    from rich.console import Console

    console = Console()
    console.print("[green]✓ All dependencies installed successfully![/green]")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

Run: `uv run test_imports.py`

## Deliverables Checklist

- [ ] `pyproject.toml` with all dependencies
- [ ] Directory structure created:
  - [ ] `src/hvas_mini/`
  - [ ] `data/memories/`
  - [ ] `logs/runs/`
  - [ ] `docs/`
- [ ] `.env.example` with all configuration options
- [ ] `src/hvas_mini/__init__.py` with version and docstring
- [ ] Dependencies successfully installed via `uv sync`
- [ ] Import test passes
- [ ] Clean worktree (no errors)

## Acceptance Criteria

1. ✅ Running `uv sync` installs all dependencies without errors
2. ✅ All directories created and accessible
3. ✅ `.env.example` includes all configuration from spec
4. ✅ Python can import all required packages
5. ✅ Ready for other branches to build on this foundation

## Notes

- This branch is BLOCKING - no other branches can proceed until complete
- Use `uv` exclusively (not pip or poetry)
- Do NOT commit `.env` files with actual keys
- Ensure Python 3.11+ is used
- Keep directory structure flat under `src/hvas_mini/`

## Testing

```bash
# From worktrees/project-foundation
uv run test_imports.py
```

Expected output: Green checkmark with success message

## Next Steps

After completion, this branch can be merged to main, and branches 2-3 can proceed in parallel:
- feature/state-management
- feature/memory-system
