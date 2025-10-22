# LEAN V3: Hierarchical Coordinator Architecture Implementation Guide

**Date**: 2025-10-22
**Status**: ✅ Fully Implemented
**Version**: V3.0.0

---

## Overview

LEAN V3 implements the **hierarchical coordinator architecture** that was originally documented but not implemented in V2. This guide provides a comprehensive reference for the V3 system architecture, components, and usage.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Implementation Details](#implementation-details)
4. [Usage Guide](#usage-guide)
5. [Configuration Options](#configuration-options)
6. [Comparison: V2 vs V3](#comparison-v2-vs-v3)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### 3-Layer Hierarchical Design

```
┌──────────────────────────────────────────────────────────────┐
│                    LAYER 1: COORDINATOR                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  CoordinatorAgentV2                                    │ │
│  │  - Tavily Research                                     │ │
│  │  - Context Synthesis & Distribution                    │ │
│  │  - Output Aggregation                                  │ │
│  │  - Quality Critique & Revision Requests                │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                            ↓ ↓ ↓
┌──────────────────────────────────────────────────────────────┐
│                 LAYER 2: CONTENT AGENTS                      │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ IntroAgent  │  │  BodyAgent  │  │ ConclusionAgent│        │
│  │   Pool (5)  │  │  Pool (5)   │  │   Pool (5)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓              ↓ (optional)        ↓                │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                 LAYER 3: SPECIALIST AGENTS                   │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Researcher  │  │ FactChecker │  │   Stylist   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

### Workflow Flow

```
START
  ↓
[1. RESEARCH]        ← Coordinator researches topic via Tavily
  ↓
[2. DISTRIBUTE]      ← Coordinator synthesizes & distributes context
  ↓
[3. INTRO]           ← IntroAgent generates with coordinator context
  ↓
[4. BODY]            ← BodyAgent generates (with optional specialist support)
  ↓
[5. CONCLUSION]      ← ConclusionAgent generates with full context
  ↓
[6. AGGREGATE]       ← Coordinator aggregates outputs
  ↓
[7. CRITIQUE]        ← Coordinator critiques quality
  ↓
[REVISION LOOP?]     ← If quality < threshold & revisions < max
  │                     YES: feedback → back to [3. INTRO]
  ↓ NO
[8. EVALUATE]        ← ContentEvaluator scores sections
  ↓
[9. EVOLVE]          ← Store patterns, trigger pool evolution
  ↓
END
```

---

## Key Components

### 1. CoordinatorAgentV2

**Location**: `src/lean/coordinator_agent.py`

**Responsibilities**:
- Research topics using Tavily API
- Synthesize research into section-specific contexts
- Distribute context to content agents
- Aggregate outputs from content agents
- Critique quality and request revisions

**Key Methods**:

```python
# Research topic
research_results = coordinator.research_topic(
    topic="AI in Healthcare",
    max_results=5,
    search_depth="advanced"
)

# Synthesize research into contexts
synthesis = coordinator.synthesize_research(
    research_results=research_results,
    reasoning_patterns=patterns,
    domain_knowledge=knowledge
)

# Critique aggregated output
critique = coordinator.critique_output(
    intro=intro_text,
    body=body_text,
    conclusion=conclusion_text,
    topic=topic,
    research_context=research_results
)
```

**Inheritance**: Extends `BaseAgentV2`, has full reasoning pattern memory and evolution support

### 2. Specialist Agents

**Location**: `src/lean/specialist_agents.py`

#### ResearcherAgent
- Deep research and evidence validation
- Source discovery and credibility assessment
- Knowledge gap identification

```python
research = researcher.research_claim(
    claim="AI improves diagnostic accuracy by 20%",
    content_context=body_context
)
```

#### FactCheckerAgent
- Claim verification against research
- Error detection and correction suggestions
- Accuracy scoring

```python
fact_check = fact_checker.check_content(
    content=body_text,
    research_context=research_summary
)
```

#### StylistAgent
- Clarity and readability enhancement
- Style refinement
- Tone consistency checking

```python
improvements = stylist.improve_style(
    content=body_text,
    target_tone="professional"
)
```

**Usage**: All specialists extend `BaseAgentV2` and support reasoning pattern evolution

### 3. PipelineV3

**Location**: `src/lean/pipeline_v3.py`

**Features**:
- Hierarchical workflow orchestration
- Tavily research integration
- Specialist invocation mechanism
- Revision loop with coordinator critique
- Agent pool evolution
- Configurable features (research, specialists, revision)

**Initialization**:

```python
pipeline = PipelineV3(
    reasoning_dir="./data/reasoning",
    shared_rag_dir="./data/shared_rag",
    domain="Healthcare",
    population_size=5,
    evolution_frequency=10,
    enable_research=True,      # Tavily integration
    enable_specialists=True,   # Layer 3 specialists
    enable_revision=True,      # Revision loop
    max_revisions=2           # Max revision iterations
)
```

**Execution**:

```python
final_state = await pipeline.generate(
    topic="The Future of AI in Medicine",
    generation_number=1
)
```

---

## Implementation Details

### Graph Structure (LangGraph)

```python
# V3 Pipeline Graph
START → research → distribute → intro → body → conclusion →
aggregate → critique → [revise OR evaluate] → evolve → END

# Conditional revision loop
critique → should_revise() → {
    "revise": revise_node → intro (loop back),
    "evaluate": evaluate_node → evolve → END
}
```

### Coordinator Research Phase

```python
async def _research_node(self, state: BlogState) -> BlogState:
    """Coordinator researches topic using Tavily."""
    research_results = self.coordinator.research_topic(
        topic=state['topic'],
        max_results=5,
        search_depth='advanced'
    )
    state['research_results'] = research_results
    return state
```

### Context Distribution

```python
async def _distribute_node(self, state: BlogState) -> BlogState:
    """Coordinator synthesizes and distributes context."""
    synthesis = self.coordinator.synthesize_research(
        research_results=state['research_results'],
        reasoning_patterns=coordinator_patterns,
        domain_knowledge=domain_knowledge
    )

    # Distribute section-specific contexts
    state['intro_coordinator_context'] = synthesis['intro_context']
    state['body_coordinator_context'] = synthesis['body_context']
    state['conclusion_coordinator_context'] = synthesis['conclusion_context']
    return state
```

### Specialist Invocation

```python
# In body_node, optional specialist support
if self.enable_specialists and self.specialists:
    specialist_context = await self._invoke_specialists(
        topic=topic,
        content_context=coordinator_context
    )

    # Researcher provides evidence
    research = self.specialists['researcher'].research_claim(
        claim=f"Research insights for: {topic}",
        content_context=content_context
    )
```

### Revision Loop

```python
def _should_revise(self, state: BlogState) -> str:
    """Conditional edge: decide whether to revise."""
    critique = state.get('coordinator_critique', {})
    revision_needed = critique.get('revision_needed', False)
    revision_count = state.get('revision_count', 0)

    if revision_needed and revision_count < self.max_revisions:
        state['revision_count'] = revision_count + 1
        return "revise"

    return "evaluate"
```

---

## Usage Guide

### Basic Usage

1. **Set up environment**:

```bash
# .env file
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_tavily_key_here  # For research

# Optional V3 settings
ENABLE_SPECIALISTS=true
ENABLE_REVISION=true
MAX_REVISIONS=2
```

2. **Run V3 pipeline**:

```bash
# Default config
python main_v3.py

# Custom config
python main_v3.py --config healthcare_study
```

3. **Output**:
```
LEAN V3: Lamarck Evolutionary Agent Network
Hierarchical Coordinator Architecture

Architecture:
  Layer 1: Coordinator (research, orchestration, critique)
  Layer 2: Content Agents (intro, body, conclusion)
  Layer 3: Specialist Agents (researcher, fact-checker, stylist)

V3 Features:
  ✅ Tavily Research: ✅
  ✅ Specialist Agents: enabled
  ✅ Revision Loop: enabled (max 2 revisions)

Generation 1/20
Topic: The Future of Artificial Intelligence
...
```

### Programmatic Usage

```python
from src.lean.pipeline_v3 import PipelineV3
import asyncio

async def generate_content():
    # Initialize V3 pipeline
    pipeline = PipelineV3(
        domain="Technology",
        population_size=5,
        evolution_frequency=5,
        enable_research=True,
        enable_specialists=True,
        enable_revision=True
    )

    # Generate content
    result = await pipeline.generate(
        topic="The Impact of Quantum Computing",
        generation_number=1
    )

    # Access results
    print("INTRO:", result['intro'])
    print("BODY:", result['body'])
    print("CONCLUSION:", result['conclusion'])

    # Coordinator critique
    critique = result.get('coordinator_critique', {})
    print("Overall Score:", critique.get('scores', {}).get('overall', 0))
    print("Feedback:", critique.get('feedback', ''))

    # Evaluation scores
    print("Intro Score:", result['scores']['intro'])
    print("Body Score:", result['scores']['body'])
    print("Conclusion Score:", result['scores']['conclusion'])

asyncio.run(generate_content())
```

---

## Configuration Options

### Environment Variables

```bash
# Core settings
ANTHROPIC_API_KEY=your_key_here
MODEL_NAME=claude-3-5-sonnet-20241022
BASE_TEMPERATURE=0.7

# Tavily Research
TAVILY_API_KEY=your_tavily_key_here
TAVILY_MAX_RESULTS=5
TAVILY_SEARCH_DEPTH=advanced

# V3 Features
ENABLE_SPECIALISTS=true  # Enable Layer 3 specialist agents
ENABLE_REVISION=true     # Enable revision loop
MAX_REVISIONS=2          # Maximum revision iterations

# Memory settings
MAX_REASONING_RETRIEVE=5
MAX_KNOWLEDGE_RETRIEVE=3
MEMORY_SCORE_THRESHOLD=7.0

# Evolution settings
EVOLUTION_LEARNING_RATE=0.1

# Visualization
ENABLE_VISUALIZATION=true
```

### YAML Configuration

V3 uses the same experiment configuration format as V2:

```yaml
# config/experiments/my_experiment.yml
experiment:
  name: "My V3 Experiment"
  description: "Testing hierarchical architecture"
  population_size: 5
  evolution_frequency: 5
  total_generations: 20

research:
  enabled: true
  max_results: 5
  search_depth: "advanced"

topic_blocks:
  - name: "AI Topics"
    generation_range: [1, 20]
    topics:
      - title: "AI in Healthcare"
        keywords: ["AI", "healthcare", "diagnosis"]
      - title: "AI Ethics"
        keywords: ["ethics", "AI", "responsibility"]
```

---

## Comparison: V2 vs V3

| Feature | V2 (Flat) | V3 (Hierarchical) |
|---------|-----------|-------------------|
| **Architecture** | Flat pipeline | 3-layer hierarchical |
| **Coordinator** | ❌ No coordinator | ✅ CoordinatorAgentV2 |
| **Research** | ❌ Not integrated | ✅ Tavily research |
| **Specialists** | ❌ Not implemented | ✅ 3 specialist agents |
| **Revision Loop** | ❌ Linear only | ✅ Coordinator critique loop |
| **Context Distribution** | Mechanical (ContextManager) | Agent-driven (Coordinator) |
| **Workflow** | intro → body → conclusion | research → distribute → agents → critique |
| **Evolution** | ✅ Pool-based | ✅ Pool-based (same) |
| **Reasoning Patterns** | ✅ Inherited | ✅ Inherited (same) |
| **Shared RAG** | ✅ Yes | ✅ Yes (same) |
| **Performance** | Faster (fewer nodes) | Slower (more comprehensive) |
| **Content Quality** | Good | Potentially better (research + critique) |
| **Complexity** | Lower | Higher |
| **Use Case** | Rapid evolution experiments | High-quality content generation |

### When to Use V2

- Focus is on **reasoning pattern evolution** research
- Speed is important (fewer LLM calls)
- Don't need external research integration
- Simpler architecture is preferred
- Running many short experiments

### When to Use V3

- Focus is on **content quality**
- Need **real-time research** integration
- Want **specialist support** (fact-checking, style)
- Need **revision loop** for quality assurance
- Willing to trade speed for quality
- Testing hierarchical coordination benefits

---

## Performance Considerations

### LLM Call Comparison

**V2 Pipeline** (per generation):
- Intro: 1 call
- Body: 1 call
- Conclusion: 1 call
- **Total: 3 calls**

**V3 Pipeline** (per generation, all features enabled):
- Research: 0 calls (Tavily API)
- Distribute (synthesis): 1 call
- Intro: 1 call
- Body: 1 call
  - Researcher: 1 call (if specialists enabled)
- Conclusion: 1 call
- Critique: 1 call
- Revisions (if needed): up to 3 calls × max_revisions
- **Total: 6-12+ calls**

### Optimization Strategies

1. **Disable Specialists** if not needed:
```python
pipeline = PipelineV3(enable_specialists=False)  # Saves ~1 call
```

2. **Disable Revision Loop** for speed:
```python
pipeline = PipelineV3(enable_revision=False)  # Saves 0-6 calls
```

3. **Reduce Max Revisions**:
```python
pipeline = PipelineV3(max_revisions=1)  # Limit revision overhead
```

4. **Disable Research** if domain knowledge sufficient:
```python
pipeline = PipelineV3(enable_research=False)  # Saves Tavily + synthesis call
```

**Minimal V3** (speed-optimized):
```python
pipeline = PipelineV3(
    enable_research=False,
    enable_specialists=False,
    enable_revision=False
)
# ~5 calls per generation (similar to V2 + coordinator overhead)
```

---

## Troubleshooting

### Common Issues

#### 1. Tavily API Errors

**Symptom**: Research phase fails
```
[Coordinator] Tavily research error: API key invalid
```

**Solution**:
- Check `TAVILY_API_KEY` in `.env`
- Sign up at https://tavily.com for API key
- Or disable research: `enable_research=False`

#### 2. Specialist Import Errors

**Symptom**:
```
ModuleNotFoundError: No module named 'lean.specialist_agents'
```

**Solution**:
```bash
# Ensure you're in project root
python -m pip install -e .

# Or use uv
uv sync
```

#### 3. Revision Loop Not Triggering

**Symptom**: No revisions despite low scores

**Check**:
- `ENABLE_REVISION=true` in `.env`
- Coordinator critique returns `revision_needed: YES`
- `revision_count < max_revisions`

**Debug**:
```python
# In critique output
critique = state['coordinator_critique']
print("Revision needed:", critique.get('revision_needed'))
print("Revision count:", state.get('revision_count', 0))
```

#### 4. Performance Issues

**Symptom**: V3 very slow

**Solutions**:
- Disable unused features (specialists, revision)
- Reduce `max_revisions`
- Use faster model: `MODEL_NAME=claude-3-haiku-20240307`
- Reduce `MAX_REASONING_RETRIEVE` and `MAX_KNOWLEDGE_RETRIEVE`

### Debugging Tips

1. **Enable Verbose Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check State Logs**:
```python
final_state = await pipeline.generate(topic="...")
for log in final_state['stream_logs']:
    print(log)
```

3. **Inspect Coordinator Outputs**:
```python
# Research results
research = final_state.get('research_results', {})
print("Found sources:", len(research.get('results', [])))

# Synthesis
synthesis = final_state.get('coordinator_synthesis', {})
print("Intro context:", synthesis.get('intro_context', '')[:200])

# Critique
critique = final_state.get('coordinator_critique', {})
print("Scores:", critique.get('scores'))
print("Feedback:", critique.get('feedback'))
```

---

## Summary

LEAN V3 successfully implements the hierarchical coordinator architecture with:

✅ **Coordinator Agent** - Research, distribution, critique
✅ **Specialist Agents** - Research, fact-checking, style support
✅ **Tavily Integration** - Real-time research
✅ **Revision Loop** - Quality assurance through critique
✅ **Pool Evolution** - Same evolutionary learning as V2
✅ **Reasoning Patterns** - Full inheritance and evolution support

**Files Created**:
- `src/lean/coordinator_agent.py` - CoordinatorAgentV2 class
- `src/lean/specialist_agents.py` - 3 specialist agent classes
- `src/lean/pipeline_v3.py` - Hierarchical pipeline orchestration
- `main_v3.py` - Entry point for V3 experiments

**Trade-offs**:
- **V2**: Faster, simpler, focused on evolution research
- **V3**: Slower, more complex, focused on content quality with research integration

Both architectures coexist - V2 for rapid evolution experiments, V3 for high-quality content generation with comprehensive support.

---

**Next Steps**:
1. Run V3 experiments: `python main_v3.py`
2. Compare V2 vs V3 output quality
3. Tune V3 features based on needs
4. Consider hybrid approaches (V2 + selective V3 features)

**Documentation**:
- V3 Implementation: This guide
- Gap Analysis: `docs/architecture-implementation-gap.md`
- V2 Documentation: `CLAUDE.md`, `README.md`
- Configuration: `docs/yaml-configuration-guide.md`
