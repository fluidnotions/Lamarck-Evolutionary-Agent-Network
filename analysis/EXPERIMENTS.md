# Running LEAN Experiments

## Quick Start

```bash
# Run with default configuration
uv run python main_v2.py

# Run with custom experiment config
uv run python main_v2.py --config my_experiment

# Show available options
uv run python main_v2.py --help
```

## Entry Points Explained

LEAN has a **single main entry point** with clear internal routing:

### Primary Entry Point

**`main_v2.py:27`** - Main experiment launcher
- Loads YAML configuration from `config/experiments/`
- Initializes PipelineV2 with evolution parameters
- Runs N generations with evolutionary learning
- Displays results and statistics
- **Use this for:** Running experiments, testing evolution, generating content

## Internal Architecture (For Understanding)

These are **not** entry points but key internal components:

- **`pipeline_v2.py:35`** - PipelineV2 class implementing 8-step cycle
- **`agent_pool.py`** - Agent pool management with M2 evolution
- **`config_loader.py:71`** - YAML config parser (load_config function)

## Configuration System

### Experiment Configuration

Create YAML files in `config/experiments/`:

```yaml
# config/experiments/my_experiment.yml
experiment:
  name: "My Experiment"
  description: "Testing something specific"

  population_size: 3        # Agents per role
  evolution_frequency: 5    # Evolve every N generations
  total_generations: 20     # Total generations to run

  reasoning_dir: "./data/reasoning"
  shared_rag_dir: "./data/shared_rag"
  domain: "General"

topic_blocks:
  - name: "Topic Group 1"
    generation_range: [1, 5]
    topics:
      - title: "First Topic"
        keywords: ["keyword1", "keyword2"]
      - title: "Second Topic"
        keywords: ["keyword3", "keyword4"]
```

### Agent Prompts

Edit `config/prompts/agents.yml` to customize agent behavior:

```yaml
intro:
  system_prompt: |
    You are an Introduction Agent...
  reasoning_focus: "engagement, clarity, preview"

body:
  system_prompt: |
    You are a Body Agent...
  reasoning_focus: "detail, evidence, structure"

conclusion:
  system_prompt: |
    You are a Conclusion Agent...
  reasoning_focus: "synthesis, impact, closure"
```

## Running Experiments

### Basic Experiment

```bash
# Run default 20-generation experiment
uv run python main_v2.py
```

**What happens:**
1. Loads `config/experiments/default.yml`
2. Creates 3 agents per role (intro, body, conclusion)
3. Runs 20 generations with evolution every 5 gens
4. Stores reasoning patterns in `data/reasoning/`
5. Stores domain knowledge in `data/shared_rag/`
6. Displays scores, patterns used, and evolution statistics

### Custom Experiment

```bash
# Create custom config
cat > config/experiments/healthcare.yml << 'EOF'
experiment:
  name: "Healthcare AI Experiment"
  population_size: 5
  evolution_frequency: 3
  total_generations: 15

topic_blocks:
  - name: "Medical AI"
    generation_range: [1, 15]
    topics:
      - title: "AI in Diagnostics"
      - title: "Machine Learning for Drug Discovery"
      - title: "Neural Networks in Radiology"
EOF

# Run it
uv run python main_v2.py --config healthcare
```

### With Tavily Research

Enable web research for richer content:

```bash
# Set API key in .env
echo "TAVILY_API_KEY=your_key" >> .env

# Enable in config
cat >> config/experiments/research.yml << 'EOF'
research:
  enabled: true
  max_results: 5
  search_depth: "advanced"
EOF

uv run python main_v2.py --config research
```

## Understanding the Output

### Generation Output

```
Generation 5/20: AI in Healthcare Applications

📊 Scores:
  Intro: 7.5
  Body: 8.2
  Conclusion: 7.8
  Average: 7.83

🧠 Memory Usage:
  Reasoning patterns used: 12
  Domain knowledge retrieved: 8

🧬 EVOLUTION EVENT!
  Selected parents: Agent #2 (fitness: 8.1), Agent #5 (fitness: 7.9)
  Compacted patterns: 15 → 10 per parent
  Created offspring: Agent #6, Agent #7, Agent #8
```

**Key Metrics:**
- **Scores (0-10)**: Content quality evaluation
- **Patterns used**: How many reasoning traces retrieved
- **Knowledge used**: Domain facts from SharedRAG
- **Evolution events**: When population reproduces

### Final Summary

```
Evolutionary Pool Statistics:
  Intro Pool: Generation 4, Avg fitness: 7.8
  Body Pool: Generation 4, Avg fitness: 8.1
  Conclusion Pool: Generation 4, Avg fitness: 7.6

Shared RAG:
  Total knowledge: 156 entries
  By source: intro: 52, body: 68, conclusion: 36

Evolutionary Learning Analysis:
  Generation 1 avg: 6.2
  Generation 20 avg: 7.9
  Total improvement: +1.7 points ✅
```

## Experiment Types

### Transfer Learning Test

Test if reasoning patterns help on similar topics:

```yaml
topic_blocks:
  # Block 1: Learn baseline patterns
  - name: "AI Basics"
    generation_range: [1, 5]
    topics:
      - title: "Machine Learning Fundamentals"
      - title: "Neural Networks Explained"

  # Block 2: Test transfer (evolution happens at gen 5)
  - name: "AI Applications"
    generation_range: [6, 10]
    topics:
      - title: "ML in Healthcare"
      - title: "Neural Nets for Diagnosis"
```

**Expected:** Improvement in Block 2 scores due to inherited patterns.

### Evolution Frequency Test

Compare different evolution schedules:

```bash
# Frequent evolution (every 2 gens)
evolution_frequency: 2

# Rare evolution (every 10 gens)
evolution_frequency: 10
```

**Hypothesis:** Frequent evolution = faster adaptation but less diversity.

### Population Size Test

Test scaling:

```bash
# Small population
population_size: 2

# Large population
population_size: 10
```

**Hypothesis:** Larger populations = more diversity but slower best agent emergence.

## Data Management

### Generated Data

After running experiments:

```
data/
├── reasoning/          # Reasoning pattern vectors
│   ├── intro/
│   ├── body/
│   └── conclusion/
└── shared_rag/         # Domain knowledge vectors
    └── chroma.sqlite3
```

### Cleaning Data

```bash
# Start fresh experiment
rm -rf data/reasoning/* data/shared_rag/*
uv run python main_v2.py
```

### Preserving Experiment Results

```bash
# Save experiment data
mv data/reasoning data/reasoning_experiment1
mv data/shared_rag data/shared_rag_experiment1

# Or use custom dirs in config
reasoning_dir: "./data/exp1/reasoning"
shared_rag_dir: "./data/exp1/shared_rag"
```

## Testing

### Run All Tests

```bash
# Full test suite
uv run pytest

# V2 pipeline tests only
uv run pytest tests/test_pipeline_v2.py -v

# M2 evolution tests
uv run pytest tests/test_agent_pool.py tests/test_evolution_integration.py -v
```

### Quick Smoke Test

```bash
# Test config loading
uv run python -c "from src.lean.config_loader import load_config; print(load_config('default')[0].name)"

# Test imports
uv run python -c "from src.lean.pipeline_v2 import PipelineV2; print('✅ Imports OK')"
```

## Troubleshooting

### Common Issues

**1. "ANTHROPIC_API_KEY not set"**
```bash
# Copy example and add key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

**2. "Config file not found"**
```bash
# List available configs
ls config/experiments/

# Use correct name without .yml
uv run python main_v2.py --config default
```

**3. "Module 'lean' not found"**
```bash
# Run with uv (handles package imports)
uv run python main_v2.py
```

**4. "GPU OOM" or "CUDA errors"**
```bash
# Force CPU for embeddings (already configured)
# Check: src/lean/shared_rag.py uses cpu_count=1
# Embedder runs on CPU to avoid GPU memory issues
```

## Advanced Usage

### Programmatic Experiments

```python
# custom_experiment.py
import asyncio
from src.lean.pipeline_v2 import PipelineV2
from src.lean.config_loader import load_config

async def main():
    exp_config, _ = load_config("my_experiment")

    pipeline = PipelineV2(
        reasoning_dir=exp_config.reasoning_dir,
        shared_rag_dir=exp_config.shared_rag_dir,
        domain=exp_config.domain,
        population_size=exp_config.population_size,
        evolution_frequency=exp_config.evolution_frequency
    )

    topics = exp_config.get_all_topics()
    results = []

    for gen, topic in enumerate(topics, 1):
        state = await pipeline.generate(topic, gen)
        results.append(state['scores'])

    return results

if __name__ == "__main__":
    results = asyncio.run(main())
    print(f"Average score: {sum(r['body'] for r in results) / len(results)}")
```

### Batch Experiments

```bash
# Run multiple configs
for config in experiment1 experiment2 experiment3; do
  echo "Running $config..."
  uv run python main_v2.py --config $config

  # Save results
  mv data/reasoning "results/${config}_reasoning"
  mv data/shared_rag "results/${config}_rag"
done
```

## Research Questions

Use LEAN to test:

1. **Does Lamarckian inheritance work?**
   - Compare Gen 1 vs Gen 20 scores
   - Expected: Improvement as patterns accumulate

2. **Do patterns transfer to similar topics?**
   - Use topic blocks with related content
   - Expected: Better scores on Block 2 after learning Block 1

3. **What's the optimal evolution frequency?**
   - Try freq=2, 5, 10
   - Measure: convergence speed vs final quality

4. **How does population size affect learning?**
   - Try size=2, 5, 10
   - Measure: diversity vs best agent quality

5. **Which selection strategy works best?**
   - Edit src/lean/agent_pool.py to swap strategies
   - Try: tournament, rank_based, fitness_proportionate

## Next Steps

1. **Run baseline experiment**: `uv run python main_v2.py`
2. **Analyze results**: Check data/reasoning/ for stored patterns
3. **Modify config**: Create custom experiment in config/experiments/
4. **Compare strategies**: Test different evolution parameters
5. **Record findings**: Use this directory (analysis/) for results

---

**For questions or issues:** Check README.md or CLAUDE.md for architecture details
