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

Instead of committing to one compaction approach, we're **testing multiple strategies in parallel** to empirically determine which inheritance mechanism works best.

### Three Baseline Strategies

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
- **Streamlit**: Dashboard for monitoring (Milestone 5)

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

### Implementation Roadmap

**Milestone 1 (Core System)**: Agent pools, fitness tracking, context distribution, memory inheritance structure
**Milestone 2 (Evolution)**: Memory compaction, reproduction, population dynamics
**Milestone 3 (Strategies)**: Compaction strategy abstraction, 3 baseline implementations
**Milestone 4 (Experimentation)**: 100-gen runs, statistical analysis, lineage tracking
**Milestone 5 (Enhancement)**: Dashboard, search integration, meta-evolution of compaction strategies

**Total timeline:** 4 weeks

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

**Current version** runs basic learning demonstration (5 topics, single agent per role).
**Next version** (Milestone 1-4) will run full evolutionary experiments.

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

## Current Status

**Phase:** Proof of concept complete
**Next:** Milestone 1 implementation (agent pools, context distribution)

**Proof of concept demonstrates:**
- ✅ Individual agent memory (ChromaDB)
- ✅ Parameter evolution (temperature adjustment)
- ✅ Transfer learning (semantic memory retrieval)
- ✅ Real-time visualization

**Milestone 1-4 will implement:**
- Agent populations (5 per role)
- Memory inheritance (inherited + personal memories)
- Memory compaction strategies (score-weighted, diversity-based, usage-based)
- Reproduction with knowledge transfer
- 100-generation experiments
- Lineage tracking and statistical comparison

---

## For Potential Collaborators

If you're reading this because you're considering joining:

**What this project is:**
- An empirical test of hybrid evolutionary learning
- A platform for comparing genetic algorithm strategies
- Research into geometric agent coordination
- A learning environment (I'm learning, you'd be learning)

**What this project isn't:**
- A production system (it's deliberately simplified)
- A guaranteed success (it might not work)
- A defined roadmap (research adapts)

**What I value:**
- Intellectual honesty ("I don't know" > speculation)
- First-principles thinking (challenge assumptions)
- Experimental rigor (measure, don't guess)
- Willingness to fail (most experiments fail—that's research)

**The interesting questions:**
- Does Lamarckian inheritance work for AI agents?
- Which memory compaction strategy produces best outcomes?
- How much inherited knowledge is optimal? (Too little = wasted potential, too much = noise)
- Does knowledge accumulation across generations beat single-agent learning?
- Can compaction strategies themselves evolve?
- Does geometric coordination actually help?

If testing those questions interests you, dive in.

---

## Documentation

- **`README.md`**: This file (research overview)
- **`docs/technical.md`**: Implementation details, setup, architecture
- **`docs/NEXT_ITERATION.md`**: Design decisions and evolutionary strategy framework
- **`docs/IMPLEMENTATION_ORCHESTRATION.md`**: Milestone breakdown, branch structure
- **`CLAUDE.md`**: AI assistant guide

---

## Final Thoughts

This is research, not engineering. The goal is to test whether **Lamarckian memory inheritance**—where agents pass learned knowledge to offspring—produces demonstrable improvement over time.

The hypothesis: prompts are the problem (that's why we need RAG). So keep prompts stable and evolve the knowledge base instead.

Maybe it works. Maybe it doesn't. Either way, we'll have empirical answers instead of speculation.

**Questions? Think the hypothesis is wrong? Want to suggest experiments?**

Open an issue. Let's discuss.

---

**HVAS Mini** — Evolution meets memory. Let's see what emerges.

---

## Technical Details

For installation, configuration, architecture documentation, and customization guides:

→ **See [docs/technical.md](docs/technical.md)**

**Quick Start:**
```bash
uv sync              # Install dependencies
uv run python main.py  # Run proof of concept (5 generations)
```

**Requirements:** Python 3.11+, Anthropic API key, uv package manager
