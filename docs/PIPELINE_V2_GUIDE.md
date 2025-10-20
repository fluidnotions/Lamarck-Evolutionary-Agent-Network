# Pipeline V2 Guide
**Date**: 2025-10-20
**Status**: ✅ Production Ready

---

## Overview

Pipeline V2 implements the complete 8-step learning cycle with reasoning pattern architecture. It replaces the old content-based memory system with cognitive pattern evolution.

**Key Feature**: Agents inherit HOW they think, not WHAT they produce.

---

## Quick Start

### Basic Usage

```python
import asyncio
from lean.pipeline_v2 import PipelineV2

async def main():
    # Initialize pipeline
    pipeline = PipelineV2(
        reasoning_dir="./data/reasoning",
        shared_rag_dir="./data/shared_rag",
        domain="Technology"
    )

    # Run a generation
    final_state = await pipeline.generate(
        topic="Understanding Neural Networks",
        generation_number=1
    )

    # Check results
    print(f"Intro score: {final_state['scores']['intro']}")
    print(f"Reasoning used: {final_state['reasoning_patterns_used']}")

asyncio.run(main())
```

### Command Line

```bash
# Run V2 pipeline with default settings
python main_v2.py

# Run demo (5 generations with progress tracking)
python examples/pipeline_v2_demo.py
```

---

## Architecture

### 8-Step Learning Cycle

Each generation executes all 8 steps:

```
┌─────────────────────────────────────────────────────┐
│ STEP 1: START                                       │
│ - Agent initializes with inherited reasoning       │
│ - Personal patterns from previous generations      │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│ STEP 2: PLAN (retrieve_similar_reasoning)          │
│ - Query reasoning patterns by similarity           │
│ - Weight by score (50/50 similarity/quality)       │
│ - Returns: Top 5 relevant cognitive strategies     │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│ STEP 3: RETRIEVE (shared_rag.retrieve)             │
│ - Query shared knowledge base                      │
│ - Returns: Top 3 relevant domain facts/content    │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│ STEP 4: CONTEXT (context_manager.assemble_context) │
│ - 40% Hierarchy (task description)                 │
│ - 30% High-credibility agents (top performers)     │
│ - 20% Diversity (random low-performer)             │
│ - 10% Same-role peer                               │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│ STEP 5: GENERATE (generate_with_reasoning)         │
│ - LLM generates with <think>/<final> tags         │
│ - <think>: Reasoning trace (cognitive strategy)   │
│ - <final>: Output content (what user sees)        │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│ STEP 6: EVALUATE (ContentEvaluator)                │
│ - Score output quality (0-10 scale)               │
│ - Multiple criteria (engagement, clarity, etc.)   │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│ STEP 7: STORE (store_reasoning_and_output)         │
│ - Store reasoning pattern (ALL, no threshold)     │
│ - Store output in shared RAG (only if ≥8.0)       │
│ - Update fitness tracking                         │
└───────────────┬─────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────┐
│ STEP 8: EVOLVE (M2 - future)                       │
│ - Selection: Choose best reasoners as parents     │
│ - Compaction: Distill reasoning patterns          │
│ - Reproduction: Create offspring with inheritance │
└─────────────────────────────────────────────────────┘
```

**Status**: Steps 1-7 fully implemented. Step 8 planned for M2.

---

## Components

### PipelineV2 Class

Main orchestrator for the V2 architecture.

```python
class PipelineV2:
    def __init__(
        reasoning_dir: str,
        shared_rag_dir: str,
        agent_ids: Optional[Dict[str, str]] = None,
        domain: str = "General"
    )
```

**Parameters**:
- `reasoning_dir`: Where to store per-agent reasoning patterns
- `shared_rag_dir`: Where to store shared domain knowledge
- `agent_ids`: Custom IDs for agents (defaults to `agent_1`)
- `domain`: Domain category for fitness tracking

**Key Methods**:
- `generate(topic, generation_number)`: Run one generation
- `get_agent_stats()`: Get agent statistics
- `get_shared_rag_stats()`: Get shared RAG statistics
- `get_context_flow_stats()`: Get context distribution metrics

---

### Agent Nodes

Each agent node (`_intro_node`, `_body_node`, `_conclusion_node`) executes steps 2-5:

```python
async def _intro_node(self, state: BlogState) -> BlogState:
    # STEP 2: Retrieve reasoning patterns
    reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(...)

    # STEP 3: Retrieve domain knowledge
    domain_knowledge = agent.shared_rag.retrieve(...)

    # STEP 4: Assemble context
    reasoning_context = self.context_manager.assemble_context(...)

    # STEP 5: Generate with reasoning
    result = agent.generate_with_reasoning(...)

    # Store in state
    state['intro'] = result['output']
    state['intro_reasoning'] = result['thinking']

    # Prepare for storage (executed in evolve node)
    agent.prepare_reasoning_storage(...)

    return state
```

---

### State Management

Extended `BlogState` with reasoning fields:

```python
class BlogState(TypedDict):
    # ... existing fields ...

    # Reasoning patterns (V2)
    intro_reasoning: str          # <think> content from intro
    body_reasoning: str           # <think> content from body
    conclusion_reasoning: str     # <think> content from conclusion
    reasoning_patterns_used: Dict[str, int]  # {role: count}
    domain_knowledge_used: Dict[str, int]    # {role: count}
    generation_number: int        # Generation sequence number
```

---

## Configuration

### Environment Variables

```bash
# Reasoning pattern retrieval
MAX_REASONING_RETRIEVE=5          # Max patterns to retrieve
INHERITED_REASONING_SIZE=100      # Max inherited from parents

# Shared RAG
SHARED_RAG_MIN_SCORE=8.0          # Quality threshold
MAX_KNOWLEDGE_RETRIEVE=3          # Max knowledge items

# Storage
REASONING_DIR=./data/reasoning    # Reasoning pattern storage
SHARED_RAG_DIR=./data/shared_rag  # Shared knowledge storage

# Visualization
ENABLE_VISUALIZATION=true          # Show Rich terminal UI

# Model
MODEL_NAME=claude-3-5-sonnet-20241022  # Claude model for generation
```

---

## Usage Patterns

### Single Generation

```python
pipeline = PipelineV2()

final_state = await pipeline.generate(
    topic="Machine Learning Basics",
    generation_number=1
)

print(f"Scores: {final_state['scores']}")
print(f"Reasoning used: {final_state['reasoning_patterns_used']}")
```

### Multi-Generation Experiment

```python
pipeline = PipelineV2(domain="ML")

topics = [
    "Neural Networks Intro",
    "Deep Learning Fundamentals",
    "CNN Architectures"
]

for gen, topic in enumerate(topics, start=1):
    state = await pipeline.generate(topic, generation_number=gen)
    print(f"Gen {gen}: Avg score = {sum(state['scores'].values()) / 3:.1f}")

# Check learning progress
stats = pipeline.get_agent_stats()
print(f"Intro agent: {stats['intro']['reasoning_patterns']} patterns")
```

### Custom Agent IDs

```python
# For agent pools or reproduction
pipeline = PipelineV2(
    agent_ids={
        'intro': 'gen2_parent1_child3',
        'body': 'gen2_parent2_child1',
        'conclusion': 'gen2_parent1_child2'
    }
)
```

---

## Testing

### Run Tests

```bash
# Non-API tests (initialization, structure)
pytest tests/test_pipeline_v2.py -k "initialization or agent_pools" -v

# API tests (requires ANTHROPIC_API_KEY)
pytest tests/test_pipeline_v2.py -v
```

### Available Tests

1. `test_pipeline_v2_initialization`: Pipeline setup
2. `test_pipeline_v2_agent_pools`: Agent pool structure
3. `test_pipeline_v2_single_generation`: One generation (requires API)
4. `test_pipeline_v2_multi_generation`: Pattern accumulation (requires API)
5. `test_pipeline_v2_context_distribution`: Context manager (requires API)
6. `test_pipeline_v2_quality_threshold`: Shared RAG threshold (requires API)

---

## Observability

### Agent Statistics

```python
stats = pipeline.get_agent_stats()

# Returns:
{
    'intro': {
        'agent_id': 'intro_agent_1',
        'role': 'intro',
        'task_count': 5,
        'avg_fitness': 8.2,
        'reasoning_patterns': 5,
        'personal_patterns': 5,
        'inherited_patterns': 0
    },
    # ... body, conclusion ...
}
```

### Shared RAG Statistics

```python
rag_stats = pipeline.get_shared_rag_stats()

# Returns:
{
    'total_knowledge': 12,
    'by_source': {
        'manual': 2,
        'generated': 10
    }
}
```

### Context Flow Statistics

```python
context_stats = pipeline.get_context_flow_stats()

# Returns:
{
    'diversity_score': 0.75,  # 0.0-1.0
    'unique_sources': 3,
    'total_sources': 4
}
```

---

## Comparison: Old vs. V2

### Old Pipeline

```python
from lean.pipeline import HVASMiniPipeline

pipeline = HVASMiniPipeline(persist_directory="./data/memories")
final_state = await pipeline.generate(topic="AI")

# What it did:
# - Stored full content in MemoryManager
# - Retrieved content examples
# - No reasoning externalization
# - No context distribution
```

### V2 Pipeline

```python
from lean.pipeline_v2 import PipelineV2

pipeline = PipelineV2(
    reasoning_dir="./data/reasoning",
    shared_rag_dir="./data/shared_rag"
)
final_state = await pipeline.generate(
    topic="AI",
    generation_number=1
)

# What it does:
# - Stores reasoning patterns (cognitive strategies)
# - Retrieves similar reasoning approaches
# - <think>/<final> tag externalization
# - 40/30/20/10 context distribution
# - Quality-gated shared knowledge
```

---

## Migration Path

### Option 1: Parallel Usage (Recommended)

Keep old pipeline for existing experiments, use V2 for new ones:

```python
# Old experiments
from lean.pipeline import HVASMiniPipeline
old_pipeline = HVASMiniPipeline()

# New experiments
from lean.pipeline_v2 import PipelineV2
new_pipeline = PipelineV2()
```

### Option 2: Full Migration

Use `main_v2.py` instead of `main.py`:

```bash
# Old
python main.py

# New
python main_v2.py
```

---

## Examples

### Example 1: Basic Usage

See: `examples/pipeline_v2_demo.py`

Demonstrates:
- 5 generations with progress tracking
- Reasoning pattern accumulation
- Shared RAG growth
- Learning improvement measurement

### Example 2: Custom Configuration

```python
pipeline = PipelineV2(
    reasoning_dir="./experiments/exp1/reasoning",
    shared_rag_dir="./experiments/exp1/shared_rag",
    agent_ids={
        'intro': 'exp1_intro_1',
        'body': 'exp1_body_1',
        'conclusion': 'exp1_conclusion_1'
    },
    domain="Science"
)

# Seed shared RAG
shared_rag = pipeline.agents['intro'].shared_rag
shared_rag.store(
    content="Quantum mechanics describes behavior at atomic scales.",
    metadata={'topic': 'quantum', 'domain': 'Science'},
    source='manual'
)

# Run generations
for i in range(10):
    await pipeline.generate(
        topic=f"Quantum Topic {i}",
        generation_number=i+1
    )
```

---

## Performance

### Typical Timings (Claude 3.5 Sonnet)

- Single agent generation: ~3-5 seconds
- Full pipeline (3 agents): ~10-15 seconds
- 5-generation experiment: ~60-90 seconds

### Storage

- Reasoning patterns: ~1KB per pattern
- Shared RAG entry: ~500B per entry
- 100 generations: ~500KB reasoning + ~50KB shared RAG

---

## Troubleshooting

### Issue 1: No reasoning patterns retrieved

**Symptom**: `reasoning_patterns_used` always 0

**Solution**: Agents need at least 1 generation before patterns available. Check:
```python
stats = pipeline.get_agent_stats()
print(stats['intro']['reasoning_patterns'])  # Should be > 0 after gen 1
```

### Issue 2: Shared RAG not growing

**Symptom**: `total_knowledge` stays constant

**Solution**: Scores may be below 8.0 threshold. Check:
```python
print(final_state['scores'])  # Need scores >= 8.0
```

Lower threshold for testing:
```python
os.environ['SHARED_RAG_MIN_SCORE'] = '7.0'
```

### Issue 3: Context distribution empty

**Symptom**: `reasoning_context` always empty

**Solution**: First generation has no prior reasoning. Check generation 2+:
```python
# Gen 1: No context (first)
# Gen 2: Should have intro reasoning from gen 1
# Gen 3: Should have intro+body reasoning
```

---

## Next Steps

### Implemented (M1):

- ✅ Three-layer architecture
- ✅ Reasoning pattern storage
- ✅ Shared RAG with quality threshold
- ✅ 8-step cycle (steps 1-7)
- ✅ Context distribution (40/30/20/10)
- ✅ BaseAgentV2 with <think>/<final> tags

### Planned (M2):

- ⏳ STEP 8: Evolution (selection, compaction, reproduction)
- ⏳ Agent pools with population management
- ⏳ Reasoning pattern compaction strategies
- ⏳ Parent selection algorithms
- ⏳ Inheritance with pattern mixing

---

## API Reference

See:
- `src/lean/pipeline_v2.py` - Pipeline implementation
- `src/lean/base_agent_v2.py` - Agent classes
- `src/lean/reasoning_memory.py` - Reasoning pattern storage
- `src/lean/shared_rag.py` - Shared knowledge base
- `src/lean/context_manager.py` - Context distribution

---

## Support

- **Documentation**: `docs/MIGRATION_GUIDE.md`
- **Examples**: `examples/pipeline_v2_demo.py`
- **Tests**: `tests/test_pipeline_v2.py`
- **Session Summary**: `docs/brainstorming/2025-10-20-SESSION-COMPLETE-SUMMARY.md`

---

**Status**: Production ready. All core components tested and validated.

🚀 **Ready for multi-generation experiments and M2 evolution implementation!**
