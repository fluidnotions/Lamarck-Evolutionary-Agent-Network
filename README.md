# HVAS Mini - A Research Exploration into Hierarchical Agent Learning

> **"I have no idea if this will work, but that's the point of research."**

This is an experiment in getting AI agents to actually *learn* from their past successes—not through fine-tuning, not through prompting tricks, but through something closer to how we learn: by remembering what worked before and applying it to similar problems later.

---

## The Core Question

**Can AI agents with their own memory and the ability to self-adjust become measurably better at tasks over time?**

Not "better" in the sense of more tokens or more context. Better in the sense of: *"Last time I wrote an intro like this, it scored 9/10, so let me try that pattern again."*

---

## The Thinking Process

### The Problem I'm Exploring

Traditional AI agent systems are stateless. Every run starts from scratch. They're like amnesiacs—brilliant, articulate amnesiacs who forget everything the moment the conversation ends.

But what if they could remember? Not just cache responses, but actually:
1. Store what worked well (based on measurable outcomes)
2. Retrieve relevant past successes when facing similar tasks
3. Adjust their behavior based on performance feedback

### The Hypothesis (That Might Be Wrong)

If each agent maintains its own memory of successful outputs (RAG-style) and can evolve its parameters based on performance scores, we should see:
- **Transfer learning**: Performance improvements on similar (not identical) tasks
- **Emergent optimization**: Self-adjusting parameters converging to optimal values
- **Specialization**: Each agent developing its own "style" based on what works for it

*Could be complete bullshit. That's what the experiment is for.*

### Why Hierarchical?

Because flat peer-to-peer pipelines don't reflect how actual work gets done. Real systems have structure:
- A **coordinator** that understands the high-level goal
- **Content specialists** that handle major pieces
- **Domain experts** that provide specialized input

The hierarchy allows for:
- **Context distribution**: High-level intent flows down
- **Result aggregation**: Detailed outputs bubble up
- **Iterative refinement**: The coordinator can request revisions based on quality gates
- **Semantic filtering**: Agents receive context weighted by relevance to their role

Again, this might not work. But it's worth exploring.

---

## What Actually Happens

### The 3-Layer Architecture

```
Layer 1: Coordinator
         ├─ Parses intent
         ├─ Distributes context to children
         ├─ Critiques output quality
         └─ Requests revisions if needed
              ↓ context flows down
Layer 2: Content Agents (Intro, Body, Conclusion)
         ├─ Receive filtered context from coordinator
         ├─ Generate their section
         ├─ Query specialists for expertise
         └─ Aggregate specialist input
              ↓ context flows down
Layer 3: Specialists (Researcher, Fact-Checker, Stylist)
         ├─ Provide deep domain expertise
         ├─ No children (leaf nodes)
         └─ Results flow back up
              ↑ results aggregate up
```

### The Learning Cycle

For each agent, each generation:
1. **Retrieve**: Query its own memory bank for past successful outputs (semantic similarity)
2. **Generate**: Create content informed by those examples + weighted context from parent
3. **Evaluate**: Get scored (0-10) on multiple quality factors
4. **Store**: If score ≥ 7.0, save this output as a new example for future use
5. **Evolve**: Adjust parameters (temperature, etc.) based on rolling performance average

### Multi-Pass Refinement

The coordinator can run multiple passes (up to 3 by default):
- First pass: Generate initial content
- Coordinator critiques: "Body is too short", "Intro lacks hook"
- Second pass: Content agents revise based on feedback
- If quality threshold met (avg confidence ≥ 0.8): Stop early
- Otherwise: Continue up to max passes

### Semantic Distance Weighting

Not all context is equally relevant. The system uses hand-crafted semantic vectors to:
- Calculate "distance" between agents (e.g., researcher is close to fact-checker, far from stylist)
- Filter context based on distance (closer = more context shared)
- Weight aggregation by semantic relevance

*Is this the right approach? Don't know yet. Testing it.*

---

## The Experiment

The demo runs 5 blog post generations:
1. **"introduction to machine learning"** - Baseline (no memories yet)
2. **"machine learning applications"** - Similar topic (should reuse memories from #1)
3. **"python programming basics"** - New topic (different domain)
4. **"python for data science"** - Similar to #3 (should reuse those memories)
5. **"artificial intelligence ethics"** - New topic

**What I'm looking for**:
- Do topics 2 and 4 score higher than topics 1 and 3? (Transfer learning)
- Do agent parameters (temperature) converge to stable values? (Emergent optimization)
- Do memory retrieval counts correlate with better outputs? (Memory effectiveness)

**Expected Results**:
- ~0.5-1.0 point score improvement on similar topics (if the hypothesis holds)
- Temperature values stabilizing after 3-4 generations
- Memory banks accumulating 2-4 high-quality examples per agent

**If it doesn't work**:
- Maybe the scoring heuristics are bad
- Maybe semantic vectors are nonsense
- Maybe the whole concept is flawed

That's fine. That's research.

---

## Why This Matters (If It Works)

If agents can demonstrably learn from experience:
- **Cheaper**: Less need for massive context windows stuffed with examples
- **Faster**: Retrieval from vector DB is faster than processing 50k token prompts
- **Specialized**: Each agent develops domain-specific expertise
- **Adaptive**: System improves with use, not just with retraining

But these are hypotheticals. The prototype is designed to test if the core mechanisms even function.

---

## Technology Stack

*For those who care about the implementation:*

- **LangGraph**: Workflow orchestration (bidirectional flows, state management)
- **Anthropic Claude**: LLM for content generation
- **ChromaDB**: Vector database for RAG memory (separate collection per agent)
- **sentence-transformers**: Text embeddings for semantic similarity
- **Rich**: Terminal UI for watching the learning happen in real-time

---

## Running It Yourself

```bash
# Prerequisites
# - Python 3.11+
# - Anthropic API key
# - uv package manager

# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the experiment
uv run python main.py
```

You'll see:
- Live visualization of agent execution
- Memory retrieval logs (when agents pull past examples)
- Quality scores for each generation
- Parameter evolution (temperature changes)
- Final statistics showing whether learning occurred

Expected runtime: ~5-10 minutes (5 generations × 3 agents + multi-pass refinement)

---

## What to Look For

### Metrics to Watch

1. **Score Progression**: Do later similar topics score higher?
2. **Memory Accumulation**: Does each agent build up a library of successful patterns?
3. **Parameter Convergence**: Do temperatures stop bouncing around?
4. **Retrieval Patterns**: Are memories actually being reused?
5. **Confidence Trends**: Does the system become more confident over time?

### The Real Test

After generation 5, compare:
- **Topic 1 score vs Topic 2 score** (ML baseline vs ML with memory)
- **Topic 3 score vs Topic 4 score** (Python baseline vs Python with memory)

If Topic 2 and Topic 4 consistently score higher, the hypothesis has legs. If not, back to the drawing board.

---

## Limitations & Unknowns

I'm deliberately keeping this simple to isolate what works and what doesn't:

**Known Limitations**:
- Heuristic scoring (not LLM-based evaluation)
- Hand-crafted semantic vectors (not learned)
- Small memory banks (no forgetting mechanism yet)
- Fixed 3-layer hierarchy (not dynamic)

**Unknown Unknowns**:
- Does semantic distance actually help, or is it just noise?
- Is 7.0 the right memory threshold, or should it be 8.0? 6.0?
- Are 3 passes enough for refinement, or should it be adaptive?
- Do the semantic vectors reflect actual semantic relationships?
- **Are trust weights and graph mutation redundant?** The system has both M2 (trust-based weighting - agents trust each other based on performance) and M4 (meta-agent that can remove/restructure agents). If trust weights can drop to ~0, that functionally achieves the same thing as removing an agent. Do we need both mechanisms, or is one sufficient? This might be over-engineering the problem.

These are research questions, not bugs. The prototype is designed to make these questions testable.

---

## What's Next

If the core mechanisms show promise:
1. **LLM-based evaluation**: Replace heuristics with actual quality assessment
2. **Learned semantic vectors**: Replace hand-crafted vectors with learned embeddings
3. **Memory consolidation**: Add forgetting, clustering, importance weighting
4. **Dynamic hierarchy**: Let agents spawn specialists on demand
5. **Cross-agent memory**: Selective sharing of successful patterns
6. **Real-world tasks**: Move beyond blog generation to code, research, analysis

But first, does the basic loop even work? That's what this prototype tests.

---

## For Potential Team Members

If you're reading this because you're considering joining:

**What I value**:
- Intellectual honesty (saying "I don't know" is encouraged)
- First-principles thinking (challenging assumptions, not just optimizing)
- Experimental rigor (testing hypotheses, not just building features)
- Willingness to fail (most research leads nowhere, and that's fine)

**What this project is**:
- An experiment in agent learning mechanics
- A testbed for measuring whether certain intuitions hold up
- A learning environment (I'm learning, you'd be learning, we'd all be learning)

**What this project isn't**:
- A production system (it's deliberately simplified)
- A sure thing (it might all be wrong)
- A well-defined roadmap (research doesn't work that way)

If that sounds interesting, the codebase is designed to be hackable. Dive in, break things, test ideas.

---

## Project Structure

```
hvas-mini/
├── main.py                          # Entry point: 5-generation experiment
├── pyproject.toml                   # uv configuration
├── .env.example                     # Environment template
├── README.md                        # This file
├── CLAUDE.md                        # AI assistant guide
│
├── src/hvas_mini/                   # Core implementation
│   ├── state.py                     # State definitions (BlogState, HierarchicalState)
│   ├── memory.py                    # MemoryManager (ChromaDB wrapper)
│   ├── agents.py                    # BaseAgent + Intro/Body/Conclusion agents
│   ├── evaluation.py                # ContentEvaluator (heuristic scoring)
│   ├── evolution.py                 # Parameter adjustment logic
│   ├── visualization.py             # StreamVisualizer (Rich terminal UI)
│   ├── pipeline.py                  # HVASMiniPipeline (LangGraph orchestration)
│   │
│   ├── hierarchy/                   # M6-M9: Hierarchical structure
│   │   ├── structure.py             # AgentHierarchy (3-layer definition)
│   │   ├── coordinator.py           # CoordinatorAgent (Layer 1)
│   │   ├── specialists.py           # Researcher/FactChecker/Stylist (Layer 3)
│   │   ├── executor.py              # HierarchicalExecutor (bidirectional flow)
│   │   ├── semantic.py              # Semantic distance calculations
│   │   └── factory.py               # Agent instantiation
│   │
│   ├── memory/                      # M3: Memory decay
│   │   ├── decay.py                 # DecayCalculator, MemoryPruner
│   │   └── __init__.py
│   │
│   ├── weighting/                   # M2: Trust-based weighting
│   │   ├── trust_manager.py         # TrustManager (confidence weighting)
│   │   ├── weight_updates.py        # Weight adjustment logic
│   │   └── __init__.py
│   │
│   ├── orchestration/               # M1: Async execution
│   │   ├── async_coordinator.py     # Concurrent agent execution
│   │   └── __init__.py
│   │
│   └── meta/                        # M4: Graph optimization
│       ├── meta_agent.py            # MetaAgent (performance analysis)
│       ├── metrics_monitor.py       # MetricsMonitor
│       ├── graph_mutator.py         # GraphMutator
│       └── __init__.py
│
├── tests/                           # Test suite
│   ├── test_state.py
│   ├── test_memory.py
│   ├── test_memory_decay.py
│   ├── test_agent_weighting.py
│   ├── test_meta_agent.py
│   ├── test_async_orchestration.py
│   ├── test_hierarchical_structure.py
│   ├── test_bidirectional_flow.py
│   ├── test_closed_loop_refinement.py
│   └── test_semantic_distance.py
│
└── docs/                            # Implementation notes
    ├── extending-agents.md
    ├── custom-evaluation.md
    ├── langgraph-patterns.md
    └── technical.md
```

---

## Configuration

Key knobs you can turn (`.env` file):

```bash
# Memory threshold: How good does output need to be to remember it?
MEMORY_SCORE_THRESHOLD=7.0    # 0-10 scale

# Evolution: How aggressively do agents adjust their parameters?
EVOLUTION_LEARNING_RATE=0.1   # 0.0-1.0

# Quality threshold: When does multi-pass refinement stop early?
QUALITY_THRESHOLD=0.8         # 0.0-1.0

# Max passes: How many refinement iterations before giving up?
MAX_PASSES=3                  # Default

# Base temperature: Starting point for all agents
BASE_TEMPERATURE=0.7          # 0.0-1.0
```

Tweak these and see what breaks (or improves).

---

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_hierarchical_structure.py -v

# Run with coverage
uv run pytest --cov=src/hvas_mini
```

Tests cover:
- Hierarchy structure (parent-child relationships, layer organization)
- Bidirectional flow (context distribution, result aggregation)
- Closed-loop refinement (multi-pass execution, quality gates)
- Semantic distance (vector calculations, context filtering)
- Memory operations (storage, retrieval, thresholds)
- Parameter evolution (temperature adjustments, convergence)

---

## Documentation

- **`CLAUDE.md`**: Guide for AI assistants working on this codebase
- **`docs/`**: Implementation patterns and customization guides

---

## Final Thoughts

This is research, not engineering. The goal isn't to build a polished product—it's to test whether a set of ideas about agent learning hold up under scrutiny.

Maybe it works. Maybe it doesn't. Either way, we'll learn something.

**Questions? Suggestions? Think I'm completely wrong?**

Open an issue. Let's discuss.

---

**Happy Exploring. 🧠🔬**
