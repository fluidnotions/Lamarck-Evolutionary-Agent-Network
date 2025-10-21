# Human-in-the-Loop (HITL) Guide

## Overview

The Human-in-the-Loop feature enables interactive oversight and feedback during evolutionary learning experiments. Instead of running fully automated experiments, you can pause execution at key points to review, edit, and approve agent outputs and decisions.

## Why Use HITL?

**Benefits**:
- **Quality Control**: Review and edit agent outputs before scoring
- **Manual Validation**: Provide expert scores instead of relying solely on automatic evaluation
- **Evolution Oversight**: Approve or reject evolution decisions
- **Pattern Curation**: Review reasoning patterns before they're stored for inheritance
- **Learning**: Understand what agents are doing and why
- **Debugging**: Identify issues in real-time during experiments

**Use Cases**:
- Initial development and testing
- High-stakes applications requiring human validation
- Research experiments where you want to understand agent behavior
- Creating high-quality training data from agent outputs
- Fine-tuning evolution parameters based on real-time feedback

## Quick Start

### Basic Usage

```python
from src.lean.human_in_the_loop import HumanInTheLoop, ReviewPoint

# Create HITL interface
hitl = HumanInTheLoop(
    enabled=True,
    review_points=[ReviewPoint.CONTENT.value, ReviewPoint.SCORING.value]
)

# Review content
result = hitl.review_content(
    role='intro',
    content=agent_output,
    topic='AI Ethics',
    generation=5
)

if result.approved:
    final_content = result.content  # May be edited
else:
    # Content rejected - regenerate or handle accordingly
    pass

# Manual scoring
score = hitl.manual_score(
    role='body',
    content=agent_output,
    auto_score=8.0
)
```

### Environment Variables

The easiest way to enable HITL is through environment variables:

```bash
# Enable HITL
export ENABLE_HITL=true

# Specify review points (comma-separated)
export HITL_REVIEW_POINTS=content,scoring

# Run experiment
python main_v2.py
```

Or inline:
```bash
ENABLE_HITL=true HITL_REVIEW_POINTS=content,scoring python main_v2.py
```

### Configuration Options

| Environment Variable | Values | Default | Description |
|---------------------|--------|---------|-------------|
| `ENABLE_HITL` | true/false | false | Enable/disable HITL globally |
| `HITL_REVIEW_POINTS` | content,scoring,evolution,patterns | all | Which stages to pause for review |
| `HITL_AUTO_APPROVE` | true/false | false | Skip reviews (dry-run mode) |
| `HITL_VERBOSE` | true/false | true | Show detailed information |

## Review Points

### 1. Content Review (`content`)

Pause after each agent generates content to review and optionally edit it.

**When it pauses**: After agent generates intro/body/conclusion content
**What you can do**:
- ✓ Approve as-is
- ✏️ Edit the content
- ✗ Reject (request regeneration)

**Example**:
```
┌─ Content Review - INTRO ──────────────────────────┐
│ Generation: 5 | Topic: AI Ethics                  │
└───────────────────────────────────────────────────┘

┌─ Intro Output ────────────────────────────────────┐
│ Artificial intelligence raises profound ethical    │
│ questions about autonomy, privacy, and fairness... │
└───────────────────────────────────────────────────┘

Options:
  1. Approve as-is
  2. Edit content
  3. Reject (regenerate)

Choose action [1/2/3]: _
```

**Use when**:
- You want to ensure quality before scoring
- Building a curated dataset
- Testing new prompts or agents

### 2. Manual Scoring (`scoring`)

Provide manual scores instead of or in addition to automatic evaluation.

**When it pauses**: Before automatic scoring is applied
**What you can do**:
- Accept automatic score
- Provide manual score (0-10)

**Example**:
```
┌─ Manual Scoring - BODY ───────────────────────────┐

┌─ Content to Score ────────────────────────────────┐
│ Machine learning algorithms excel at pattern...   │
└───────────────────────────────────────────────────┘

Automatic score: 8.0/10

Provide manual score? [y/N]: y
Enter score (0-10) [8.0]: 8.5

✓ Score: 8.5/10
```

**Use when**:
- Automatic evaluation doesn't match your criteria
- You want to apply domain expertise
- Building labeled data for evaluation model training

### 3. Evolution Review (`evolution`)

Approve or reject evolution events before population replacement.

**When it pauses**: Before evolution replaces current population
**What you can do**:
- Review parent statistics
- See offspring count
- Approve or reject evolution

**Example**:
```
┌─ Evolution Review - INTRO ────────────────────────┐
│ Generation: 10                                    │
└───────────────────────────────────────────────────┘

┌─ Parents ─────────────────────────────────────────┐
│ Agent ID            Fitness  Patterns             │
│ intro_gen9_agent1   7.8      15                   │
│ intro_gen9_agent2   7.5      12                   │
│ intro_gen9_agent3   7.2      18                   │
└───────────────────────────────────────────────────┘

Offspring: 5 new agents

Approve this evolution? [Y/n]: _
```

**Use when**:
- You want to control when evolution occurs
- Testing evolution parameters
- Debugging population dynamics

### 4. Pattern Review (`patterns`)

Review reasoning patterns before they're stored for inheritance.

**When it pauses**: Before storing reasoning pattern to memory
**What you can do**:
- Review reasoning quality
- Approve or reject storage

**Example**:
```
┌─ Pattern Review - BODY ───────────────────────────┐
│ Score: 8.5/10                                     │
└───────────────────────────────────────────────────┘

┌─ Reasoning Pattern ───────────────────────────────┐
│ Situation: Explain ML to general audience         │
│                                                    │
│ Reasoning:                                         │
│ The key challenge is balancing technical accuracy │
│ with accessibility. I should explain using        │
│ analogies familiar to readers...                  │
└───────────────────────────────────────────────────┘

Store this pattern? [Y/n]: _
```

**Use when**:
- Curating high-quality patterns for inheritance
- Debugging pattern quality issues
- Building pattern library

## Practical Examples

### Example 1: Content Quality Control

Only review and approve content, skip other stages:

```bash
ENABLE_HITL=true HITL_REVIEW_POINTS=content python main_v2.py
```

This will pause after each agent generates content, letting you:
- Fix typos or grammar issues
- Adjust tone or style
- Ensure accuracy before scoring

### Example 2: Manual Scoring Experiment

Provide expert scores for a small number of generations:

```bash
ENABLE_HITL=true HITL_REVIEW_POINTS=scoring python main_v2.py
```

Generate outputs automatically, but score them manually. Useful for:
- Evaluating new evaluation criteria
- Creating training data for automatic scorers
- Comparing automatic vs human scores

### Example 3: Supervised Evolution

Control evolution decisions:

```bash
ENABLE_HITL=true HITL_REVIEW_POINTS=evolution python main_v2.py
```

Let agents run automatically, but approve each evolution event. Useful for:
- Ensuring population diversity
- Preventing premature convergence
- Understanding evolution dynamics

### Example 4: Full Interactive Mode

Pause at every stage:

```bash
ENABLE_HITL=true python main_v2.py
```

Maximum oversight - review everything. Useful for:
- Initial development
- Understanding agent behavior
- Debugging issues

### Example 5: Dry-Run Mode

See what would be reviewed without actually pausing:

```bash
ENABLE_HITL=true HITL_AUTO_APPROVE=true python main_v2.py
```

Shows HITL panels but auto-approves everything. Useful for:
- Testing HITL integration
- Seeing what data would be available for review
- Demo/presentation mode

## Integration with Existing Code

### In `pipeline_v2.py`

```python
from src.lean.human_in_the_loop import create_hitl_from_env

class PipelineV2:
    def __init__(self, ...):
        # ... existing code ...
        self.hitl = create_hitl_from_env()

    async def _content_node(self, state: Dict):
        # ... generate content ...

        # HITL review
        if self.hitl.enabled:
            result = self.hitl.review_content(
                role=role,
                content=generated_content,
                topic=state['topic'],
                generation=state['generation']
            )

            if result.approved:
                generated_content = result.content
            else:
                # Handle rejection (regenerate, skip, etc.)
                pass

        # ... continue with scoring ...
```

### In Custom Scripts

```python
from src.lean.human_in_the_loop import HumanInTheLoop, ReviewPoint

# Configure for your use case
hitl = HumanInTheLoop(
    enabled=True,
    review_points=[ReviewPoint.SCORING.value]
)

# Your generation loop
for generation in range(num_generations):
    content = agent.generate(topic)

    # Optional: review content
    # ...

    # Manual scoring
    score = hitl.manual_score(
        role=agent.role,
        content=content,
        auto_score=evaluator.score(content)
    )

    # Store with manual score
    if score >= 7.0:
        agent.memory.store(content, score)
```

## Statistics and Reporting

View HITL statistics at any time:

```python
hitl.show_stats()
```

Output:
```
════════════════════════════════════════════════════════════
Human-in-the-Loop Statistics
════════════════════════════════════════════════════════════
Reviews Requested    15
Reviews Approved     12
Reviews Rejected     3
Content Edits        2
Manual Scores        5
Approval Rate        80.0%
════════════════════════════════════════════════════════════
```

## Tips and Best Practices

### 1. Start Small

Don't try to review everything in a 20-generation experiment:
- Start with 2-3 generations
- Review only 1-2 stages initially
- Gradually expand as you get comfortable

### 2. Use Targeted Review

Review specific stages that matter most:
- **Debugging**: `content` to see what agents generate
- **Quality**: `scoring` to validate evaluation
- **Control**: `evolution` to manage population
- **Curation**: `patterns` to build quality libraries

### 3. Combine with Logging

Enable detailed logging alongside HITL:
```bash
ENABLE_HITL=true ENABLE_LOGGING=true python main_v2.py
```

This gives you:
- Interactive review in real-time
- Complete logs for post-analysis

### 4. Use Auto-Approve for Demos

When showing the system to others:
```bash
ENABLE_HITL=true HITL_AUTO_APPROVE=true HITL_VERBOSE=true python main_v2.py
```

Shows what HITL does without interrupting flow.

### 5. Batch Similar Reviews

If reviewing 5 similar intro outputs, establish criteria:
- First 1-2: Review carefully, establish standards
- Remaining: Quick approve/reject based on established criteria
- Track patterns in rejection reasons

## Troubleshooting

### HITL Not Pausing

Check:
1. `ENABLE_HITL=true` is set
2. Review points are spelled correctly
3. Code calls HITL methods
4. `HITL_AUTO_APPROVE` is not true

### Input Not Recognized

If prompts don't accept your input:
- Check you're using correct choices (e.g., "1", "2", "3")
- Ensure terminal supports interactive input
- Try simpler terminal (avoid IDE integrated terminals)

### Slow Reviews

HITL adds overhead:
- Each review takes 10-30 seconds
- 20 generations × 3 roles × 3 reviews = 180 reviews (~30-90 minutes)
- Use fewer review points or generations

### Memory Issues

Long inputs in edit mode:
- Use `Ctrl+D` or empty line to finish input
- Don't paste extremely long text
- Consider file-based editing for major changes

## Advanced: Custom Review Logic

Extend `HumanInTheLoop` for custom behavior:

```python
class CustomHITL(HumanInTheLoop):
    def review_content(self, role, content, **kwargs):
        # Add custom pre-filtering
        if len(content) < 100:
            # Auto-reject very short content
            return ReviewResult(approved=False, feedback="Too short")

        # Call parent implementation
        return super().review_content(role, content, **kwargs)
```

## See Also

- **Demo Script**: `examples/hitl_demo.py` - Interactive demonstration
- **API Reference**: Docstrings in `src/lean/human_in_the_loop.py`
- **Pipeline Integration**: `src/lean/pipeline_v2.py` - Full implementation
- **Main Script**: `main_v2.py` - Command-line usage

---

**Status**: Feature complete, ready for testing
**Branch**: `feat/human-in-the-loop`
**Worktree**: `../lean-hitl`
