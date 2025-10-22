# LEAN Configuration Files

This directory contains YAML configuration files for experiments, agent prompts, and documentation.

## Quick Start

```bash
# Run experiment with default configuration
python main_v2.py

# Run experiment with custom configuration
python main_v2.py --config my_experiment
```

## Directory Structure

```
config/
├── experiments/     # Experiment configurations
│   └── default.yml  # Default 20-gen AI topics experiment
├── prompts/         # Agent system prompts and evaluation
│   └── agents.yml   # Prompts for all 7 agents
└── docs/            # Markdown documentation referenced by configs
    ├── coordinator-role.md
    ├── ai-fundamentals.md
    └── ...
```

## Creating Custom Experiments

### 1. Copy Default Config

```bash
cp config/experiments/default.yml config/experiments/healthcare_study.yml
```

### 2. Edit Configuration

```yaml
experiment:
  name: "Healthcare AI Study"
  description: "Testing medical AI topics"
  total_generations: 10

topic_blocks:
  - name: "Medical Applications"
    generation_range: [1, 10]
    topics:
      - title: "AI in Radiology"
        keywords: ["radiology", "imaging", "diagnosis"]
        difficulty: "advanced"
      # Add 9 more topics...
```

### 3. Run Your Experiment

```bash
python main_v2.py --config healthcare_study
```

## Tavily Research Integration

### Setup

1. Get API key from https://tavily.com
2. Add to `.env`:
   ```
   TAVILY_API_KEY=your_key_here
   ```
3. Enable in config:
   ```yaml
   research:
     enabled: true
     max_results: 5
     search_depth: "advanced"
   ```

The coordinator agent will automatically research each topic before distributing context to child agents.

## Configuration Files

### experiments/default.yml

Defines:
- Experiment metadata (name, description)
- Evolution parameters (population, frequency, generations)
- Topic blocks with generation ranges
- Tavily research settings
- Quality thresholds

### prompts/agents.yml

Defines:
- System prompts for each agent role
- Reasoning focus areas
- Evaluation criteria
- Links to detailed documentation

## Documentation

See **[docs/yaml-configuration-guide.md](../docs/yaml-configuration-guide.md)** for comprehensive guide including:

- Complete configuration reference
- Tavily research setup and usage
- Creating custom experiments
- Best practices
- Troubleshooting

## Examples

### Minimal Config (5 generations, no research)

```yaml
experiment:
  name: "Quick Test"
  total_generations: 5
  evolution_frequency: 5

topic_blocks:
  - name: "Test Topics"
    generation_range: [1, 5]
    topics:
      - title: "Topic 1"
      - title: "Topic 2"
      - title: "Topic 3"
      - title: "Topic 4"
      - title: "Topic 5"

research:
  enabled: false
```

### Research-Heavy Config

```yaml
research:
  enabled: true
  max_results: 10
  search_depth: "advanced"
  include_domains:
    - "arxiv.org"
    - "nature.com"
```

## Tips

1. **Topic Blocks**: Group related topics to test transfer learning
2. **Evolution Alignment**: Make blocks span evolution_frequency generations
3. **Research**: Enable for factual/technical topics, disable for creative writing
4. **Documentation**: Create markdown docs in `docs/` for complex topic blocks

## Need Help?

- Configuration guide: `docs/yaml-configuration-guide.md`
- Project docs: `README.md`
- Architecture: `CLAUDE.md`
