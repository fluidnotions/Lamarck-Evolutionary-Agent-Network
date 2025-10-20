# Lamarck Evolutionary Agent Network

**Evolutionary Agent Learning Research**

## The Core Question

**Can AI agents improve by inheriting their parents' reasoning patterns?**

This is a research prototype testing **Lamarckian evolution** for AI agent cognition:

- **Prompts stay frozen** (consistent behavioral interface)
- **Domain knowledge lives in shared RAG** (facts available to all agents)
- **Reasoning patterns are inherited** (successful cognitive strategies pass to offspring)
- **Selection determines what propagates** (natural selection on thinking patterns, not content)

Think of it as: **Acquired reasoning strategies become heritable DNA.**

The goal is to test whether cognitive pattern accumulation across generations produces agents that demonstrably improve at problem-solving—without the fragility of prompt engineering.

---

## Why This Matters

### The Problem: We're Evolving the Wrong Layer

Current AI agent systems break with every model update. They rely on carefully crafted text prompts that need constant rewriting. We're trying to encode *how to think* in natural language instructions, when we should be letting reasoning patterns emerge from experience.

**The insight:** Models don't think in text—they think in embeddings. Vector databases excel at finding similar reasoning structures. "Find planning sequences like mine" is exactly what embedding search was built for.

What if agents evolved their cognitive strategies through inheritance rather than prompt engineering?

### The Three-Layer Architecture

This system cleanly separates three concerns that current approaches conflate:

**1. Fixed Interface Layer (Prompts)**
```python
role_prompts = {
    'intro': "You write introductions.",
    'body': "You write body paragraphs.",
    'conclusion': "You write conclusions."
}
```
- Never changes, never mutates
- Provides stable API for the system
- Pure role definition

**2. Shared Knowledge Layer (RAG)**
- Domain facts and content
- Reference materials  
- Historical outputs
- Available to all agents equally
- Standard semantic retrieval

**3. Evolving Reasoning Layer (What Gets Inherited)**
- Planning sequences ("First establish context, then add statistics, finally pose question")
- Problem-solving steps
- Reasoning traces
- Cognitive strategies that worked
- Retrieved by structural similarity

**Result:** Systems that improve their reasoning through use, not through prompt engineering.

---

## The Evolutionary Architecture

### How Reasoning Inheritance Works

Each agent maintains two types of cognitive patterns:

**1. Inherited Reasoning** (from parents)
- Compacted/distilled planning strategies from both parents
- High-performing cognitive patterns that led to success
- Passed down through reproduction

**2. Personal Reasoning** (from experience)
- Own discovered planning sequences
- Reasoning traces with scores
- Added to inheritance pool when reproducing

**Memory structure:**
```python
class AgentMemory:
    def plan(self, task):
        # Retrieve similar successful planning sequences
        similar_plans = vector_db.search(
            query=f"task: {task}",
            filter={"score": {">": 7.0}},
            search_field="planning_steps"  # Search on reasoning, not content
        )
        
        # Synthesize new plan from successful patterns
        return self.merge_planning_patterns(similar_plans)
    
    def execute(self, plan):
        # Execute the evolved planning sequence
        result = self.follow_steps(plan)
        
        # Store the planning pattern + score (not the content)
        self.store_memory(
            planning_steps=plan,
            execution_trace=self.get_execution_details(),
            score=evaluate(result)
        )
```

### What Gets Stored: Cognitive Strategies, Not Content

Memories are **reasoning patterns**, not domain knowledge:

```python
{
    "planning_steps": [
        "First establish context with historical example",
        "Then introduce 3-5 supporting statistics", 
        "Finally pose thought-provoking question"
    ],
    "execution_trace": "Used WWII example, included 3 economic stats, asked 'What if?'",
    "context_type": "technical_audience",
    "score": 8.5,
    "retrieval_count": 12,
    "generation": 5
}
```

**Critical distinction (storage has two paths):**
- **Individual memory** (per-agent): ALL reasoning patterns stored (no threshold) → periodically compacted → best 20-30% inherited by offspring
- **Shared RAG** (global): ONLY high-quality content (score ≥8.0) stored → available to all agents as domain knowledge

This separation enables Lamarckian evolution: agents inherit their parents' **best** cognitive strategies (after forgetting unsuccessful ones) while sharing domain facts

### Evolutionary Operators

**Selection** (choose parents):
- ε-greedy, tournament, or fitness-weighted
- Based on how well their reasoning patterns perform
- High-scoring cognitive strategies reproduce more

**Reasoning Pattern Compaction** (create offspring):
```python
def reproduce(parent1, parent2):
    # Merge both parents' reasoning patterns
    combined_patterns = parent1.reasoning_patterns + parent2.reasoning_patterns
    
    # Compact to manageable size using strategy
    inherited_reasoning = compact_reasoning(combined_patterns, max_size=100)
    
    # Child gets compacted cognitive strategies, same prompt as parents
    child = Agent(
        prompt=role_prompts[parent1.role],  # Fixed
        inherited_reasoning=inherited_reasoning,  # Evolved
        shared_rag=global_knowledge_base  # Shared
    )
    return child
```

**Population Dynamics:**
- 5 agents per role (15 total)
- Evolution every 10 generations
- Add agents when reasoning diversity drops
- Remove agents when fitness < 6.0 (after 20+ tasks)

### The Learning Cycle

Each agent, each generation:

1. **Start with inheritance**: Already has proven reasoning patterns from parents
2. **Plan approach**: Query inherited + personal reasoning patterns for similar tasks
3. **Retrieve knowledge**: Get relevant facts from shared RAG (domain content)
4. **Receive context**: Get reasoning traces from other agents (40% hierarchy, 30% high-performers, 20% random, 10% peer)
5. **Generate**: Execute plan using fixed prompt + evolved reasoning + retrieved knowledge
6. **Evaluate**: Get scored by LLM on quality factors
7. **Store** (two paths):
   - Reasoning pattern → individual memory (ALL patterns, no threshold)
   - Output content → shared RAG (ONLY if score ≥8.0)
8. **Evolve** (every N generations, M2 implementation):
   - **Compaction**: Forget unsuccessful patterns (keep top 20-30%)
   - **Selection**: Best reasoners chosen as parents
   - **Reproduction**: Child inherits compacted reasoning patterns
   - **Population management**: Maintain diversity

**Key insight:** Adding a successful planning step from memory is functionally equivalent to editing a prompt to include that step, but it happens dynamically through retrieval rather than manual engineering.

---

## Memory Compaction Strategies

Instead of compacting content, we compact **reasoning patterns**. Each strategy encapsulates different approaches to cognitive inheritance:

### Three Baseline Strategies (A/B Testing)

#### Strategy A: Score-Weighted Selection
```python
def compact_reasoning(combined, max_size=100):
    # Keep reasoning patterns that led to highest scores
    return sorted(combined, key=lambda m: m.score)[-max_size:]
```
- **Hypothesis**: Quality over quantity—only pass on proven thinking patterns

#### Strategy B: Diversity Preservation
```python
def compact_reasoning(combined, max_size=100):
    # Keep diverse reasoning approaches for different problem types
    clusters = cluster_by_pattern_structure(combined)
    return [cluster.best_pattern for cluster in clusters][:max_size]
```
- **Hypothesis**: Coverage beats optimization—need different cognitive strategies

#### Strategy C: Usage-Based Retention
```python
def compact_reasoning(combined, max_size=100):
    # Keep reasoning patterns that were frequently retrieved and useful
    return sorted(combined,
                 key=lambda m: m.retrieval_count * m.score
                )[-max_size:]
```
- **Hypothesis**: Field-tested thinking beats theoretical quality

### What We're Measuring

Each strategy tracks:
- **Reasoning improvement**: How cognitive strategies evolve
- **Pattern diversity**: Variety of problem-solving approaches
- **Efficiency**: Quality gain per reasoning step
- **Specialization**: Role-specific cognitive patterns
- **Innovation rate**: Novel high-scoring reasoning sequences

---

## The Geometric Insight

Traditional prompt engineering uses discrete, textual instructions:

```python
prompt = "Always consider counterarguments in paragraph 3"  # Brittle text
```

Reasoning pattern inheritance uses continuous, geometric relationships:

```python
reasoning_vector = similar_patterns + δ(innovation)  # Semantic evolution
```

**This is the natural way to evolve cognitive strategies in embedding space.**

In this system:
- **Distance encodes similarity**: Close reasoning patterns (in embedding space) solve similar problems
- **Retrieval becomes natural**: Vector search excels at finding similar cognitive structures
- **Evolution emerges from geometry**: Successful patterns cluster and propagate

---

## Why This Design

### Clean Separation of Concerns

**Critical decision:** Keep three layers completely separate:

1. **Prompts (Interface)**: What agents should do—never changes
2. **RAG (Knowledge)**: What agents know—shared by all
3. **Reasoning (Cognition)**: How agents think—evolves through inheritance

This prevents contamination between layers and makes the system more robust.

### Why Lamarckian > Darwinian for AI Reasoning

**Darwinian evolution** (biological): Random mutations hope to improve
**Lamarckian evolution** (AI systems): Successful reasoning gets passed on directly

For AI agents, Lamarckian is superior because:
- **Direct cognitive transfer**: No need to rediscover reasoning patterns
- **Faster convergence**: Start each generation with proven thinking strategies
- **Preserves insights**: Successful cognitive approaches don't get lost
- **Natural selection still applies**: Bad reasoning leads to poor performance → no reproduction

**Example:**
```python
# Generation 1 agent discovers through experience:
personal_reasoning = {
    "pattern": ["establish_context", "add_statistics", "pose_question"],
    "score": 7.8
}

# Generation 2 inherits this as prior reasoning:
inherited_reasoning = {
    "pattern": ["establish_context", "add_statistics", "pose_question"],
    "score": 7.8,
    "generation": 1
}

# Generation 2 refines it:
personal_reasoning = {
    "pattern": ["establish_context", "add_statistics", "contrast_viewpoint", "pose_question"],
    "score": 8.6
}

# Generation 3 inherits the refined pattern
```

Cognitive strategies accumulate. Each generation reasons better.

---

## What Success Looks Like

### Measurable Outcomes (100 Generations)

- **Reasoning improvement**: Average scores increase through better cognitive strategies
- **Emergent specialization**: Roles develop distinct reasoning patterns
- **Sustained diversity**: Multiple problem-solving approaches coexist
- **Pattern effectiveness**: Retrieved reasoning correlates with performance
- **Strategy winner**: One compaction approach demonstrably outperforms

### Observable Phenomena

- **Reasoning lineages**: Successful cognitive patterns propagate across generations
- **Pattern refinement**: Later generations have more sophisticated reasoning
- **Cognitive accumulation**: 50-100 inherited + 50-150 personal patterns per agent
- **Cross-pollination**: Diverse reasoning prevents cognitive monoculture
- **Role specialization**: Intro agents think differently than Body agents

### What Failure Looks Like

- **No reasoning improvement**: Cognitive patterns don't evolve
- **Pattern convergence**: All agents think the same way
- **Noise accumulation**: Inherited patterns add confusion instead of clarity
- **Lost diversity**: Selection pressure eliminates alternative reasoning
- **No specialization**: Roles don't develop distinct cognitive strategies

---

## The Experiment

### Task: Blog Post Generation

Three agent roles working sequentially:
- **IntroAgent**: Writes introductions (evolving reasoning for hooks)
- **BodyAgent**: Writes body sections (evolving reasoning for explanation)
- **ConclusionAgent**: Writes conclusions (evolving reasoning for synthesis)

Each role has 5 competing agents with **identical prompts** but different inherited reasoning patterns.

### Test Dataset

1. **Tracing Thought: How Neural Activation Maps Reveal Machine Cognition**  
   Tests baseline reasoning about technical explanations

2. **Evolving Insight: Can AI Learn to Interpret Itself?**  
   Tests retention and reapplication of reasoning patterns

3. **Quantum Selection: What Evolution Might Look Like in Quantum AI**  
   Tests adaptation of reasoning to new domains

4. **Quantum Minds and Digital Species: Evolution Beyond Classical Computation**  
   Tests synthesis of reasoning across topics

5. **The Evolution of Understanding: From Biological Brains to Self-Explaining Machines**  
   Tests unified reasoning across all previous concepts

### Execution

Run all 3 strategies in parallel for 100 generations:
- Strategy A population (score-weighted reasoning)
- Strategy B population (diversity-preserved reasoning)
- Strategy C population (usage-based reasoning)

Same prompts, same RAG, different reasoning evolution.

---

## Technical Implementation

### Technology Stack

- **LangGraph**: Workflow orchestration
- **Anthropic Claude**: LLM for generation and evaluation
- **ChromaDB**: Vector database for reasoning patterns (separate collection per agent)
- **sentence-transformers**: Embeddings for reasoning similarity
- **Shared RAG**: Separate vector DB for domain knowledge
- **Tavily**: Web search for external knowledge
- **Streamlit**: Dashboard for monitoring evolution

### Key Architecture Points

```
┌─────────────────────────────────────────┐
│     Fixed Prompt Layer (Interface)      │
│         "You write introductions"       │
└────────────────┬────────────────────────┘
                 │
┌────────────────┼────────────────────────┐
│   Shared Knowledge Layer (RAG)          │
│   Domain facts, content, references     │
└────────────────┼────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Evolving Reasoning Layer (Inherited)   │
│  Planning patterns, cognitive strategies│
│         ┌──────────────────┐           │
│         │ Agent 1 Reasoning │           │
│         └──────────────────┘           │
│         ┌──────────────────┐           │
│         │ Agent 2 Reasoning │           │
│         └──────────────────┘           │
│              ... (5 per role)           │
└─────────────────────────────────────────┘
```

### Project Structure

```
src/lean/
├── agents.py              # BaseAgent with reasoning inheritance
├── agent_pool.py          # AgentPool for population management
├── reasoning_memory.py    # ReasoningMemory for cognitive patterns
├── shared_rag.py         # SharedRAG for domain knowledge
├── evaluation.py         # Evaluator for scoring outputs
├── pipeline.py           # LangGraph workflow orchestration
├── state.py              # BlogState and ReasoningPattern models
└── strategies/           # Compaction strategies for reasoning
```

---

## Running the Experiment

```bash
# Prerequisites
# - Python 3.11+
# - Anthropic API key
# - uv package manager

# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY

# Run experiment
uv run python main.py
```

### Key Configuration

```bash
# Reasoning Memory
INHERITED_REASONING_SIZE=100    # Max reasoning patterns inherited
PERSONAL_REASONING_SIZE=150     # Max personal patterns stored

# Evolution
EVOLUTION_FREQUENCY=10           # Generations between evolution
MIN_POPULATION=3                # Per role
MAX_POPULATION=8                # Per role

# Three-Layer Separation
USE_SHARED_RAG=true             # Separate domain knowledge
REASONING_SEARCH_ONLY=true      # Don't mix content with reasoning
PROMPTS_IMMUTABLE=true          # Never modify prompts
```

---

## What This Research Explores

### For AI Engineering

- **Stable multi-agent systems**: Fixed prompts + evolving reasoning
- **Self-improving cognition**: Reasoning patterns get better with use
- **Clean architecture**: Separation of interface, knowledge, and cognition
- **Dynamic prompt-equivalent behavior**: Without fragile prompt engineering

### For Evolutionary Computation

- **Cognitive inheritance**: Can reasoning strategies be inherited?
- **Pattern compaction**: How to distill cognitive strategies?
- **Reasoning selection**: Which patterns produce best outcomes?
- **Meta-cognition**: Can reasoning about reasoning evolve?

### For Cognitive Architecture

- **Generational learning**: Each generation thinks better
- **Semantic reasoning**: Cognitive patterns in embedding space
- **Emergent specialization**: Roles develop unique thinking styles
- **Cultural transmission**: Reasoning strategies pass through generations

---

## Current Status

**Implementation Progress:**
- ✅ Three-layer architecture fully implemented
- ✅ Reasoning pattern storage (ReasoningMemory class)
- ✅ Reasoning retrieval by structural similarity + score weighting
- ✅ Shared RAG for domain knowledge (SharedRAG class)
- ✅ BaseAgentV2 with <think>/<final> tag externalization
- ✅ Context distribution (40/30/20/10 weighted reasoning traces)
- ✅ 8-step learning cycle (steps 1-7 complete)
- ✅ Agent factory function (create_agents_v2())
- ✅ Comprehensive test suite (8/8 tests passing)
- ⏳ Pattern compaction strategies (M2 - future)
- ⏳ Evolutionary selection and reproduction (M2 - future)
- ⏳ Pipeline integration (next step)

**Working Features:**
- Fixed prompt layer (IntroAgentV2, BodyAgentV2, ConclusionAgentV2)
- Shared RAG with quality threshold (score ≥ 8.0)
- Per-agent reasoning pattern memory with inheritance support
- <think>/<final> tag parsing for reasoning externalization
- ContextManager for reasoning trace distribution
- Fitness tracking and evaluation framework
- Storage separation (reasoning vs. domain knowledge)
- One-line migration from old agents via factory function

**Status**: Core reasoning pattern architecture complete. Ready for pipeline integration and multi-generation testing.

---

## Research Objectives

This experiment investigates:
- Does reasoning pattern inheritance produce demonstrable improvement?
- Which cognitive compaction strategy works best?
- How do reasoning patterns evolve differently from content?
- Can agents develop role-specific cognitive strategies?
- Does this approach outperform traditional prompt engineering?

**Core hypothesis:** Prompts are bad at encoding how to think (that's why CoT helps). So keep prompts as simple interfaces, put content in RAG, and let reasoning patterns evolve through inheritance.

**Approach:** Empirical testing with statistical analysis. We're evolving cognition, not content.
