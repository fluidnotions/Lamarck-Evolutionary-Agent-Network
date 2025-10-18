# HVAS Mini - Evolutionary Agent Learning Research

## The Core Question

**Can AI agents improve by inheriting their parents' learned knowledge?**

This is a research prototype testing **Lamarckian evolution** for AI agents:

- **Prompts stay stable** (consistent behavior interface)
- **Memories are inherited** (successful knowledge passes to offspring)
- **Selection determines what propagates** (natural selection on knowledge, not random mutation)

Think of it as: **Acquired wisdom becomes heritable DNA.**

The goal is to test whether knowledge accumulation across generations produces agents that demonstrably improve over time—without the fragility of prompt engineering.

---

## Why This Matters

### The Problem: Agent Frameworks Are Fragile

Current AI agent systems break with every model update. They rely on carefully crafted text prompts that need constant rewriting. We're building agentic systems in the wrong language.

**Models don't think in text—they think in embeddings.** Geometric representations where meaning is encoded as position and relationships are measured by distance.

What if agents coordinated through geometry rather than instructions?

### The Hypothesis (Subject to Testing)

**Prompts are the problem, not the solution.** That's why we need RAG in the first place—prompts can't encode nuanced knowledge.

So instead of mutating prompts (perpetuating the problem), this system:
- **Keeps prompts stable**: "You are an intro writer" never changes
- **Evolves the knowledge base**: Successful memories get inherited
- **Uses natural selection**: Better knowledge → better performance → more offspring
- **Accumulates wisdom**: Each generation builds on the last

This is **Lamarckian evolution**—acquired characteristics (learned knowledge) get passed on directly. For AI systems, this should be superior to Darwinian random mutation.

**Result:** Systems that improve through use, not just through retraining.

---

## The Evolutionary Architecture

### The Learning Architecture

#### Stable Prompts (Interface Layer)

Prompts are **constants**, not variables:

```python
role_prompts = {
    'intro': "You write introductions.",
    'body': "You write body paragraphs.",
    'conclusion': "You write conclusions."
}
```

**Why stable?** Consistent behavior, no prompt engineering bloat, no mutation degradation.

#### Memory Inheritance (Knowledge Layer)

Each agent has two types of memory:

**1. Inherited Memories** (from parents)
- Compacted/distilled knowledge from both parents
- High-value insights that led to success
- Passed down through reproduction

**2. Personal Memories** (from experience)
- Own lifetime experiences
- All outputs + scores stored
- Added to inheritance pool when reproducing

**Memory structure:**
```python
class Agent:
    inherited_memories: List[Memory]  # From parents
    personal_memories: List[Memory]   # From own experience

    def all_memories(self):
        return self.inherited_memories + self.personal_memories
```

#### Evolutionary Operators

**Selection** (choose parents):
- ε-greedy, tournament, or fitness-weighted
- High-performing agents reproduce more

**Memory Compaction** (create offspring):
```python
def reproduce(parent1, parent2):
    # Merge both parents' full memory banks
    combined = parent1.all_memories() + parent2.all_memories()

    # Compact to manageable size using strategy
    inherited = compact_memories(combined, max_size=100)

    # Child gets compacted wisdom, same prompt as parents
    child = Agent(
        prompt=role_prompts[parent1.role],  # Stable
        inherited_memories=inherited         # Evolved
    )
    return child
```

**Population Dynamics:**
- 5 agents per role (15 total)
- Evolution every 10 generations
- Add agents when diversity drops
- Remove agents when fitness < 6.0 (after 20+ tasks)

### The Learning Cycle

Each agent, each generation:

1. **Start with inheritance**: Already has compacted wisdom from parents
2. **Retrieve**: Query all memories (inherited + personal) for semantically similar successes
3. **Receive Context**: Get weighted context from other agents (40% hierarchy, 30% high-performers, 20% random low-performer, 10% peer)
4. **Generate**: Create content using stable prompt + retrieved memories + external context
5. **Evaluate**: Get scored by LLM on quality factors (engagement, clarity, depth)
6. **Store**: Add output + score to personal memory bank
7. **Evolve** (every 10 generations):
   - **Selection**: Best agents chosen as parents
   - **Compaction**: Merge + distill parents' memories
   - **Reproduction**: Child inherits compacted knowledge
   - **Population management**: Add/remove agents based on performance

**Key insight:** Prompts stay stable. Memories evolve through inheritance. Natural selection determines which knowledge propagates.

---

## Memory Compaction Strategies

Instead of committing to one compaction approach, we implement the **Strategy Pattern** to test multiple inheritance mechanisms in parallel. Each strategy encapsulates different decisions about selection, compaction, and evolution.

### Three Baseline Strategies (A/B Testing)

#### Strategy A: Score-Weighted Selection
```python
def compact_memories(combined, max_size=100):
    # Keep memories that led to highest scores
    return sorted(combined, key=lambda m: m.score)[-max_size:]
```
- **Selection**: ε-greedy with high exploitation (90/10)
- **Compaction**: Pure score ranking
- **Evolution**: Slow (every 20 generations)
- **Hypothesis**: Quality over quantity—only pass on proven winners

#### Strategy B: Diversity Preservation
```python
def compact_memories(combined, max_size=100):
    # Keep diverse memories covering different topics
    clusters = cluster_by_embedding(combined)
    return [cluster.best_example for cluster in clusters][:max_size]
```
- **Selection**: Tournament (top 3 compete)
- **Compaction**: Cluster-based diversity
- **Evolution**: Fast (every 5 generations)
- **Hypothesis**: Coverage beats optimization—need examples across domains

#### Strategy C: Usage-Based Retention
```python
def compact_memories(combined, max_size=100):
    # Keep memories that were frequently retrieved and useful
    return sorted(combined,
                 key=lambda m: m.retrieval_count * m.score
                )[-max_size:]
```
- **Selection**: Fitness-proportional probability
- **Compaction**: Retrieval count × score
- **Evolution**: Adaptive (speeds up when stagnant)
- **Hypothesis**: Field-tested knowledge beats theoretical quality

### What We're Measuring

Each strategy tracks:
- **Fitness trajectory**: Average and best scores over 100 generations
- **Diversity metrics**: Population variance, unique strategies
- **Efficiency**: Fitness gain per API call
- **Specialization**: Domain-specific performance variance
- **Innovation rate**: Novel high-scoring outputs

**The experiment:** Run 100 generations for each strategy on identical task sets, then compare. Which configuration produces the best learning?

---

## The Geometric Insight

Traditional object-oriented systems use discrete, symbolic relationships:

```python
class Child extends Parent  # Hierarchical syntax
```

Vector-based systems use continuous, contextual relationships:

```python
vector(Child) = vector(Parent) + Δ(role)  # Semantic topology
```

**This is the natural generalization of inheritance for probabilistic systems.**

In HVAS:
- **Distance encodes influence**: Closer agents (in embedding space) share more context
- **Context becomes geometry**: Agents receive weighted information based on semantic proximity
- **Coordination emerges from topology**: No hardcoded rules—just navigate the meaning-space

The 40/30/20/10 context distribution isn't arbitrary—it's creating a gradient field where:
- 40% from hierarchy (structured intent)
- 30% from high-performers (proven patterns)
- 20% from random low-performers (forced diversity)
- 10% from role peers (specialist knowledge)

Information flows through the geometry. Success reshapes the landscape.

---

## Why This Design

### Context is Additive, Not Selective

**Critical decision:** High-performing agents get to *broadcast* their knowledge more widely, but they do NOT get privileged access to more information.

This prevents "rich get richer" dynamics that cause:
- Premature convergence (all agents become similar)
- Loss of diversity (alternative strategies die)
- Reduced innovation (no cross-pollination)

By keeping information access equal while varying influence, we maintain evolutionary diversity.

### Why Lamarckian > Darwinian for AI

**Darwinian evolution** (biological): Random mutations hope to stumble on improvements
**Lamarckian evolution** (AI systems): Successful learning gets passed on directly

For AI agents, Lamarckian is superior because:
- **Direct knowledge transfer**: No need to re-learn what parents already know
- **Faster convergence**: Start each generation with accumulated wisdom
- **Preserves hard-won insights**: Successful patterns don't get lost
- **Natural selection still applies**: Bad knowledge leads to poor performance → no reproduction

**Example:**
```python
# Generation 1 agent learns through experience:
personal_memory = "Questions increased engagement by 31% on technical topics"

# Generation 2 inherits this as prior knowledge:
inherited_memory = "Questions increased engagement by 31% on technical topics"

# Generation 2 builds on it:
personal_memory = "Questions + statistics in second sentence = 43% engagement"

# Generation 3 inherits both insights:
inherited_memories = [
    "Questions increased engagement by 31%",
    "Questions + statistics in second sentence = 43%"
]
```

Knowledge accumulates. Each generation starts ahead.

### Multiple Compaction Strategies = Empirical Answers

We don't *know* which memory compaction approach works best. So we test them in parallel:
- **Score-weighted**: Pass on only the best (elite inheritance)
- **Diversity-based**: Pass on diverse examples (coverage inheritance)
- **Usage-based**: Pass on field-tested knowledge (practical inheritance)

This isn't just A/B testing—it's **meta-optimization of knowledge inheritance**. The compaction strategies themselves could eventually evolve.

---

## What Success Looks Like

### Measurable Outcomes (100 Generations)

- **Fitness improvement**: Average scores increase by >0.5 points
- **Emergent specialization**: Agents develop domain-specific expertise (variance >1.0)
- **Sustained diversity**: Population doesn't collapse to single strategy (std dev >0.5)
- **Memory effectiveness**: Retrieval count correlates with performance
- **Strategy winner**: One configuration demonstrably outperforms others

### Observable Phenomena

- **Knowledge lineages**: Successful insights propagate across generations
- **Memory accumulation**: Inherited (50-100) + personal (50-150) = 100-250 total experiences per agent
- **Compaction quality**: Later generations have higher-quality inherited memories
- **Cross-pollination**: 20% diversity injection prevents echo chambers
- **Specialization divergence**: Intro agents accumulate different knowledge than Body agents

### What Failure Looks Like

- **All strategies converge**: Compaction approach doesn't matter (inheritance mechanism is irrelevant)
- **No generational improvement**: Later generations perform same as Generation 1 (memory inheritance doesn't help)
- **Memory bloat**: Inherited memories add noise instead of signal
- **Population diversity collapses**: Selection pressure too strong (premature convergence)
- **No domain specialization**: Agents don't develop distinct knowledge bases

If any of these happen, we learn what *doesn't* work. That's still valuable.

---

## The Experiment

### Task: Blog Post Generation

Three agent roles working sequentially:
- **IntroAgent**: Writes introductions
- **BodyAgent**: Writes body sections
- **ConclusionAgent**: Writes conclusions

Each role has 5 competing agents (15 total population).

### Test Dataset
 
 1. **Tracing Thought: How Neural Activation Maps Reveal Machine Cognition**  
   Baseline post exploring mechanistic interpretability — how activation pathways can be visualized and analyzed to understand “how models think.” References research like Google’s *Activation Atlas* and Anthropic’s *circuits* work.

2. **Evolving Insight: Can AI Learn to Interpret Itself?**  
   Builds on post #1 by theorizing self-interpreting agents capable of evolving their own activation tracing tools. Tests retention and reapplication of interpretability concepts.

3. **Quantum Selection: What Evolution Might Look Like in Quantum AI**  
   Shifts domain to quantum computing. Examines whether evolutionary optimization could operate within quantum latent spaces and whether “fitness” can exist in probabilistic computation.

4. **Quantum Minds and Digital Species: Evolution Beyond Classical Computation**  
   Extends #3 by blending quantum theory with evolutionary biology. Explores analogies like coherence as a survival trait and entanglement as a form of digital symbiosis.

5. **The Evolution of Understanding: From Biological Brains to Self-Explaining Machines**  
   Integrative post that synthesizes biological, quantum, and AI-evolution ideas. Tests the model’s ability to unify concepts from all previous posts into a coherent narrative about machine cognition.


### Execution

Run all 3 strategies in parallel for 100 generations:
- Strategy A population (isolated)
- Strategy B population (isolated)
- Strategy C population (isolated)

Same task sequence, same initial populations, no cross-contamination.

### Analysis

Statistical comparison:
- t-tests between strategies
- ANOVA for multi-strategy comparison
- Regression analysis (fitness vs generation)
- Diversity metrics over time
- Specialization emergence patterns

**Goal:** Identify which evolutionary configuration produces the best learning dynamics.

---

## Technical Implementation

### Technology Stack

- **LangGraph**: Workflow orchestration
- **Anthropic Claude**: LLM for generation and evaluation
- **ChromaDB**: Vector database (separate collection per agent)
- **sentence-transformers**: Embeddings for semantic similarity
- **Tavily**: Web search for external knowledge
- **Streamlit**: Dashboard for monitoring (future)

### Project Structure

```
src/hvas_mini/
├── agents.py              # BaseAgent and role-specific agents
├── agent_pool.py          # AgentPool for population management
├── memory.py              # MemoryManager with ChromaDB
├── evaluation.py          # ContentEvaluator scoring
├── pipeline.py            # LangGraph workflow orchestration
├── state.py               # BlogState and AgentMemory models
├── web_search.py          # Tavily search integration
├── hierarchy/             # Hierarchical coordination (legacy)
├── memory/                # Memory decay utilities
├── weighting/             # Trust-based weighting (legacy)
├── orchestration/         # Async coordination (legacy)
└── meta/                  # Graph optimization (legacy)

tests/
├── test_agent_pool.py           # Agent pool tests
├── test_memory.py               # Memory system tests
├── test_hierarchical_*.py       # Hierarchy tests
└── ...                          # Additional test modules
```

### Architecture

```
┌─────────────────────────────────────────┐
│     Evolutionary Strategy Manager       │
│  (Strategy A / B / C configurations)    │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┼─────────┬─────────┐
    │         │         │         │
┌───▼────┐ ┌─▼─────┐ ┌─▼─────┐ ┌─▼──────┐
│Intro×5 │ │Body×5 │ │Concl×5│ │Evaluator│
│Pool    │ │Pool   │ │Pool   │ │ (LLM)  │
└───┬────┘ └─┬─────┘ └─┬─────┘ └────────┘
    │        │         │
    ▼        ▼         ▼
[ChromaDB] [ChromaDB] [ChromaDB]
15 isolated collections
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

# Run current prototype (proof of concept)
uv run python main.py
```

**Current version** runs basic learning demonstration with evolutionary agent pools.

### Running Tests

All implementations require comprehensive unit tests in the `tests/` directory:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_memory.py -v

# Run with coverage
uv run pytest tests/ --cov=src/hvas_mini --cov-report=html
```

**Test Coverage Requirements**:
- All new classes and functions must have unit tests
- Target: >90% code coverage per module
- Test files follow naming: `test_<module_name>.py`
- Each AGENT_TASK.md includes specific test cases to implement

---

## Key Configuration

```bash
# Memory
MEMORY_SCORE_THRESHOLD=7.0      # (Deprecated in M1: store ALL experiences)
INHERITED_MEMORY_SIZE=100       # Max memories inherited from parents

# Evolution
EVOLUTION_FREQUENCY=10          # Generations between evolution cycles
MIN_POPULATION=3                # Per role
MAX_POPULATION=8                # Per role

# Strategy-specific (set by strategy)
SELECTION_METHOD=epsilon_greedy|tournament|fitness_weighted
COMPACTION_METHOD=score_weighted|diversity_based|usage_based
CONTEXT_WEIGHTS=40,30,20,10     # Hierarchy/High/Low/Peer
```

---

## What This Research Explores

### For AI Engineering

- **Stable multi-agent systems**: Coordinate through embeddings, not fragile prompts
- **Self-improving agents**: Systems that get better with use
- **Evolutionary robustness**: Population diversity prevents catastrophic failure
- **Compositional learning**: Agents as basis vectors in semantic space

### For Evolutionary Computation

- **Lamarckian AI**: Does inheritance of acquired characteristics work for agents?
- **Memory compaction**: How to distill knowledge across generations?
- **Knowledge selection**: Which compaction strategy produces best outcomes?
- **Meta-evolution**: Can compaction strategies themselves evolve?

### For Cognitive Architecture

- **Generational learning**: Each generation builds on parents' knowledge
- **Semantic coordination**: Agents navigate meaning-space, not call graphs
- **Emergent specialization**: Roles arise from accumulated knowledge, not programming
- **Cultural transmission**: Knowledge passes through generations (like human learning)

---

## Future Enhancements

### Phase 2: Neural Circuit Evolution (Planned)

The current system evolves **knowledge** (memories) while keeping model architecture fixed. A future enhancement will evolve the **neural pathways themselves** using open-source models with LoRA adapters.

#### Concept: Evolvable Model Weights

Instead of using Claude API (black box), use open models (Mistral, Llama, etc.) where you can:
- **Discover circuits**: Find which neural pathways activate for different content types
- **Inject LoRA adapters**: Add trainable parameters to specific layers
- **Evolve weights**: Apply genetic algorithms to LoRA adapter weights
- **Sexual reproduction**: Crossover + mutation of model parameters between agents

#### Architecture

```python
class NeuralEvolvableAgent:
    """Agent with evolvable neural architecture"""

    def __init__(self, base_model='mistralai/Mistral-7B-Instruct'):
        self.base_model = load_model(base_model)  # Frozen base

        # Each agent gets unique LoRA adapter
        self.lora_adapter = LoraConfig(
            r=8,  # Low rank
            target_modules=["q_proj", "v_proj"],  # Attention layers
            lora_dropout=0.1
        )

        # Circuit discovery results
        self.specialized_circuits = None

    def discover_specialization(self):
        """Find which neurons activate for agent's role"""
        from transformer_lens import HookedTransformer

        # Hook into model layers
        _, cache = self.model.run_with_cache(test_prompts)

        # Find "hot" neurons for this agent's tasks
        self.specialized_circuits = identify_active_pathways(cache)

    def reproduce_with_partner(self, partner):
        """Sexual reproduction of LoRA weights"""

        child = NeuralEvolvableAgent(self.base_model)

        # Crossover: Mix LoRA weights from both parents
        for key in self.lora_adapter.state_dict():
            parent1_weight = self.lora_adapter.state_dict()[key]
            parent2_weight = partner.lora_adapter.state_dict()[key]

            # Random crossover mask
            mask = torch.rand_like(parent1_weight) > 0.5
            child_weight = torch.where(mask, parent1_weight, parent2_weight)

            # Mutation: Small random perturbations
            if random.random() < 0.1:
                child_weight += torch.randn_like(child_weight) * 0.01

            child.lora_adapter.load_state({key: child_weight})

        # Inherit discovered circuits
        child.specialized_circuits = merge_circuits(
            self.specialized_circuits,
            partner.specialized_circuits
        )

        return child
```

#### Implementation Options

**Lightweight (Fast)**:
- **Model**: `microsoft/phi-2` (2.7B params)
- **Circuit Discovery**: Heuristic pathway evolution (test injection points)
- **Runs on**: Consumer GPU (8GB VRAM)

**Advanced (Thorough)**:
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Circuit Discovery**: Full transformer analysis with `transformer-lens`
- **Runs on**: High-end GPU (24GB VRAM)

#### Why This is Powerful

Current system: Evolve **what the model knows** (memories)
Future system: Evolve **how the model thinks** (neural weights)

**Benefits**:
- **True neural evolution**: Not just knowledge inheritance, but architectural adaptation
- **Specialization at weight level**: Agents develop unique neural pathways
- **No API costs**: Local models, full control
- **Research potential**: Explore which neural circuits matter for different tasks

#### Toggle Feature

```python
# Configuration
USE_NEURAL_EVOLUTION = False  # Default: memory-only evolution

if USE_NEURAL_EVOLUTION:
    # Use open models with circuit discovery
    agent = NeuralEvolvableAgent('mistralai/Mistral-7B')
    agent.discover_specialization()  # Slow initial setup
else:
    # Use Claude API with memory evolution (current)
    agent = ClaudeAgent()
```

**Status**: Not currently implemented. Requires significant computational resources and is beyond the scope of the initial research question. Could be explored in future work if memory inheritance experiments show promise.

---

## Current Status

**Implementation Progress:**
- ✅ Agent pool infrastructure (5 agents per role)
- ✅ Individual memory collections with inheritance support
- ✅ Web search integration (Tavily)
- ⏳ Fitness tracking and specialization detection
- ⏳ Context distribution system (40/30/20/10 weighting)
- ⏳ Evolutionary workflow integration

**Working Features:**
- Individual agent memory (ChromaDB with isolated collections)
- Transfer learning (semantic memory retrieval)
- Real-time visualization
- Web search for external knowledge

---

## Research Objectives

This experiment investigates:
- Does Lamarckian memory inheritance produce demonstrable improvement over time?
- Which memory compaction strategy (score-weighted, diversity-based, usage-based) works best?
- How much inherited knowledge is optimal before diminishing returns?
- Does knowledge accumulation across generations outperform single-agent learning?
- Can compaction strategies themselves evolve through meta-optimization?

**Core hypothesis:** Prompts are bad at encoding nuanced knowledge (that's why we need RAG). So keep prompts stable and evolve the knowledge base instead.

**Approach:** Empirical testing with statistical analysis. Both positive and negative results provide valuable data.