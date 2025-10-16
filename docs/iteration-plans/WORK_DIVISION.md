# HVAS Mini - Work Division Plan

## Overview
This document outlines the implementation strategy for the HVAS Mini Prototype, divided into 8 feature branches that can be developed in parallel or sequence based on dependencies.

## Dependency Graph

```
project-foundation (base)
    ├── state-management (parallel)
    └── memory-system (parallel)
        ├── base-agent (depends on state + memory)
        │   └── specialized-agents (depends on base-agent)
        ├── evaluation-system (depends on state)
        ├── visualization (depends on state)
        └── langgraph-orchestration (depends on ALL)
```

## Branch Breakdown

### Branch 1: `feature/project-foundation`
**Priority**: Critical - Must complete first
**Execution**: Sequential (blocking)

**Responsibilities**:
- Initialize `uv` project structure
- Setup `pyproject.toml` with all dependencies
- Create `.env.example` with all configuration options
- Create basic directory structure (`src/`, `data/`, `logs/`, `docs/`)
- Setup `.gitignore` (include `worktrees/`, `data/`, `logs/`)
- Create `__init__.py` files

**Deliverables**:
- Working uv project that installs all dependencies
- Configuration templates
- Clean project structure

---

### Branch 2: `feature/state-management`
**Priority**: High - Needed by most components
**Execution**: Parallel with memory-system

**Responsibilities**:
- Implement `src/state.py` with `BlogState` TypedDict
- Implement `AgentMemory` Pydantic model
- Add type hints and validation
- Create state initialization helpers

**Deliverables**:
- `src/state.py` with complete state definitions
- Type-safe state management

**Dependencies**: project-foundation

---

### Branch 3: `feature/memory-system`
**Priority**: High - Core HVAS concept
**Execution**: Parallel with state-management

**Responsibilities**:
- Implement `src/memory.py` with ChromaDB integration
- Setup sentence-transformers for embeddings
- Implement memory storage and retrieval functions
- Add metadata tracking
- Create memory persistence configuration

**Deliverables**:
- `src/memory.py` with RAG memory management
- ChromaDB collection management
- Embedding utilities

**Dependencies**: project-foundation

---

### Branch 4: `feature/base-agent`
**Priority**: High - Required for all agents
**Execution**: Sequential (blocks specialized-agents)

**Responsibilities**:
- Implement `src/agents.py` with `BaseAgent` abstract class
- Memory retrieval integration
- Parameter evolution logic (temperature adjustment)
- Score-based learning mechanism
- Configuration loading from .env

**Deliverables**:
- `src/agents.py` with complete BaseAgent
- `src/evolution.py` for parameter evolution logic

**Dependencies**: state-management, memory-system

---

### Branch 5: `feature/specialized-agents`
**Priority**: Medium - Application layer
**Execution**: Sequential (after base-agent)

**Responsibilities**:
- Implement `IntroAgent` in `src/agents.py`
- Implement `BodyAgent` in `src/agents.py`
- Implement `ConclusionAgent` in `src/agents.py`
- Create prompts with memory integration
- Add context passing between agents

**Deliverables**:
- Three specialized agent implementations
- Prompt templates optimized for each role

**Dependencies**: base-agent

---

### Branch 6: `feature/evaluation-system`
**Priority**: Medium - Needed for learning
**Execution**: Parallel with base-agent

**Responsibilities**:
- Implement `src/evaluation.py` with `ContentEvaluator`
- Multi-factor scoring for intro/body/conclusion
- Configurable scoring thresholds
- Score logging and tracking

**Deliverables**:
- `src/evaluation.py` with complete evaluation logic
- Scoring metrics documentation

**Dependencies**: state-management

---

### Branch 7: `feature/visualization`
**Priority**: Low - Enhancement feature
**Execution**: Parallel with other features

**Responsibilities**:
- Implement `src/visualization.py` with Rich library
- Real-time streaming visualization
- Status tables, memory panels, evolution panels
- Activity logs
- Configuration flags for enable/disable

**Deliverables**:
- `src/visualization.py` with StreamVisualizer
- Beautiful terminal UI with live updates

**Dependencies**: state-management

---

### Branch 8: `feature/langgraph-orchestration`
**Priority**: Critical - Integration layer
**Execution**: Sequential (after ALL components)

**Responsibilities**:
- Implement `main.py` with `HVASMiniPipeline`
- LangGraph StateGraph construction
- Node definitions and edge connections
- Streaming execution with astream()
- Async coordination
- Demo execution with multiple topics

**Deliverables**:
- `main.py` with complete pipeline
- Working end-to-end system
- Demo script showing learning over multiple runs

**Dependencies**: ALL previous branches

---

## Execution Strategy

### Phase 1: Foundation (Sequential)
1. project-foundation → BLOCKING

### Phase 2: Core Systems (Parallel)
2. state-management ⚡ memory-system → PARALLEL

### Phase 3: Agent Layer (Mixed)
3. base-agent → BLOCKING for specialized-agents
4. specialized-agents
5. evaluation-system ⚡ visualization → PARALLEL with agents

### Phase 4: Integration (Sequential)
6. langgraph-orchestration → BLOCKING (needs everything)

### Phase 5: Documentation (Parallel)
7. README.md ⚡ docs/ customization guides

---

## Key Patterns & Principles

### LangGraph Patterns
- Use `StateGraph` for workflow definition
- Leverage `astream()` for streaming execution
- Use `MemorySaver` for checkpointing
- Implement async nodes for parallel execution
- Use `TypedDict` for state management

### Customization Separation
- All business logic in `src/`
- Configuration via `.env` (no hardcoding)
- Customization guides in `docs/`
- Clear extension points in base classes

### uv Package Management
- Use `pyproject.toml` for all dependencies
- Lock file for reproducibility
- Virtual environment management via uv
- Fast dependency resolution

---

## Success Criteria

Each branch must:
1. ✅ Include `AGENT_TASK.md` with detailed instructions
2. ✅ Pass type checking (mypy compatible)
3. ✅ Follow the spec implementation exactly
4. ✅ Be runnable in isolation (where possible)
5. ✅ Include inline documentation
6. ✅ Be ready to merge to main

Final integration must:
1. ✅ Generate blog posts on given topics
2. ✅ Show memory retrieval and storage
3. ✅ Demonstrate parameter evolution
4. ✅ Stream execution with visualization
5. ✅ Prove learning across multiple runs
