# Scoring System Explanation

**Created**: 2025-10-22
**Branch**: feat/hierarchical-visualization

## Your Question

> "If you turn off human in the loop, does the content not get scored at all? How does it decide which to pick from the agent pool?"

## Answer: YES, Automatic Scoring Always Happens

### ContentEvaluator - The Automatic Scorer

**Location**: `src/lean/evaluation.py`

✅ **Automatic scoring ALWAYS runs** - HITL on/off doesn't affect this.

The `ContentEvaluator` scores content on **0-10 scale** using heuristics:

**Intro Scoring:**
- Base: 5.0
- +1.5 for 20-60 words
- +1.5 for topic mention
- +2.0 for engaging hooks

**Body Scoring:**
- Base: 5.0
- +2.0 for 100-300 words
- +1.5 for topic mention
- +1.0 for structure (paragraphs/bullets)
- +0.5 for depth indicators

**Conclusion Scoring:**
- Base: 5.0
- +1.5 for 30-80 words
- +1.5 for topic mention
- +1.5 for call-to-action
- +0.5 for intro callback

### How Scores Drive Evolution

**File**: `src/lean/pipeline.py:649-677`

```python
def _evolve_node(self, state: BlogState) -> BlogState:
    for role, pool in self.agent_pools.items():
        score = state['scores'].get(role, 0.0)  # ← From ContentEvaluator

        for agent in pool.agents:
            agent.record_fitness(score=score)  # ← Fitness tracking
            agent.store_reasoning_and_output(score=score)  # ← Pattern storage with quality
```

**Scores are used for:**
1. **Fitness tracking** - Agents accumulate score history
2. **Pattern storage** - Reasoning patterns tagged with quality scores
3. **Agent selection** - Tournament/rank selection uses fitness
4. **Evolution** - Parent selection based on fitness scores

### HITL Status

**Config exists** in `config/experiments/default.yml`:
```yaml
hitl:
  enabled: true
  auto_approve: false
```

**❌ Problem**: HITL is **NOT implemented** - these settings do nothing currently.

**✅ Automatic scoring works** regardless of HITL settings.

## What Would HITL Do (When Implemented)?

HITL would **augment** automatic scoring, not replace it:

```
[ContentEvaluator] → automatic_scores (0-10)
        ↓
[HITL] → Show content + auto scores to human
        ↓
     Human adjusts scores (optional)
        ↓
final_scores = human_scores OR automatic_scores
        ↓
[Evolution] uses final_scores
```

##How Agent Selection Works

### Pool Selection Process

**File**: `src/lean/agent_pool.py`

```python
def select_active_agent(self) -> BaseAgent:
    """Select agent for current generation."""
    return fitness_proportionate_selection(self.agents, self.fitness_scores)
```

**Selection Strategies** (from `src/lean/selection.py`):
1. **Tournament Selection** - Pick N agents, select best
2. **Rank Selection** - Weight by fitness rank
3. **Fitness Proportionate** - Probability based on fitness

**Current default**: Tournament selection with tournament_size=3

### Where Scores Come From

```
Generation N:
1. Agent generates content
2. ContentEvaluator scores it (0-10)
3. Score stored as fitness
4. Agent stores reasoning pattern with score

Generation N+1:
1. Pool selects active agent (using fitness scores)
2. Selected agent generates new content
3. Repeat...

Every X generations:
1. Evolution triggered
2. Select parents (using fitness)
3. Reproduce → offspring
4. Offspring inherit high-scoring patterns
```

## Experiment Logging

### NEW: run_experiment.sh

**Usage**:
```bash
./run_experiment.sh test
./run_experiment.sh default
```

**Output**: `experiment_logs/20251022_143052_test.log`

**Features**:
- Automatic timestamped logs
- Colored terminal output
- Log file size reporting
- Logs excluded from git

## Summary

| Question | Answer |
|----------|--------|
| **Does scoring happen without HITL?** | ✅ YES - ContentEvaluator always runs |
| **How are scores used?** | Fitness tracking → Agent selection → Evolution |
| **Is HITL implemented?** | ❌ NO - Config exists but not wired up |
| **Where do logs go?** | `experiment_logs/` (use run_experiment.sh) |

---

**Key Takeaway**: Automatic scoring is the foundation. HITL (when implemented) would add human judgment on top of automatic scores, not replace them.
