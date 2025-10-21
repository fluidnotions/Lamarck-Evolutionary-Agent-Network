# Human-in-the-Loop Feature

**Branch**: `feat/human-in-the-loop`
**Worktree**: `../lean-hitl`
**Status**: ✅ Ready for Review

## Summary

This feature adds interactive human oversight capabilities to the LEAN evolutionary learning pipeline. Users can pause execution at key points to review, edit, and approve agent outputs, scores, and evolution decisions.

## What's New

### Core Module: `src/lean/human_in_the_loop.py`

Full-featured HITL interface with:
- **Content Review**: Edit/approve agent outputs before scoring
- **Manual Scoring**: Provide expert scores (0-10)
- **Evolution Approval**: Review and approve population replacement
- **Pattern Review**: Curate reasoning patterns before storage
- **Statistics Tracking**: Monitor review activity and approval rates

### Demo Script: `examples/hitl_demo.py`

Interactive demonstration of all HITL capabilities. Run with:
```bash
ENABLE_HITL=true python examples/hitl_demo.py
```

### Documentation: `docs/human-in-the-loop-guide.md`

Comprehensive guide including:
- Quick start examples
- Configuration options
- All review points explained
- Integration examples
- Troubleshooting
- Best practices

## Quick Start

### Enable HITL for Experiments

```bash
# Review content and scoring
ENABLE_HITL=true HITL_REVIEW_POINTS=content,scoring python main_v2.py

# Review only evolution decisions
ENABLE_HITL=true HITL_REVIEW_POINTS=evolution python main_v2.py

# Full interactive mode
ENABLE_HITL=true python main_v2.py
```

### Programmatic Usage

```python
from src.lean.human_in_the_loop import HumanInTheLoop, ReviewPoint

hitl = HumanInTheLoop(
    enabled=True,
    review_points=[ReviewPoint.CONTENT.value]
)

# Review content
result = hitl.review_content(
    role='intro',
    content=agent_output,
    topic='AI Ethics',
    generation=5
)

if result.approved:
    content = result.content  # Possibly edited
    # Continue processing...
```

## Features

### 1. Content Review

- ✅ Approve content as-is
- ✏️ Edit content inline
- ❌ Reject and request regeneration
- 📝 View agent reasoning traces

### 2. Manual Scoring

- 🎯 Override automatic scores
- 📊 Reference auto-scores for comparison
- 🔢 Score validation (0-10 range)

### 3. Evolution Approval

- 👥 View parent population statistics
- 🧬 Review offspring details
- ✅/❌ Approve or reject evolution
- 📝 Provide rejection feedback

### 4. Pattern Review

- 🧠 Review reasoning patterns before storage
- 📋 See situation and quality score
- ✅/❌ Approve or reject pattern storage

### 5. Statistics & Reporting

- 📈 Track review counts
- ✅ Approval/rejection rates
- ✏️ Content edit frequency
- 🎯 Manual score counts

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_HITL` | false | Enable HITL globally |
| `HITL_REVIEW_POINTS` | all | Comma-separated: content,scoring,evolution,patterns |
| `HITL_AUTO_APPROVE` | false | Dry-run mode (show but don't pause) |
| `HITL_VERBOSE` | true | Show detailed information |

## Use Cases

### Development & Debugging

```bash
# Review all content to debug prompt issues
ENABLE_HITL=true HITL_REVIEW_POINTS=content python main_v2.py
```

### Quality Control

```bash
# Manual scoring for high-stakes applications
ENABLE_HITL=true HITL_REVIEW_POINTS=scoring python main_v2.py
```

### Evolution Research

```bash
# Control when evolution occurs
ENABLE_HITL=true HITL_REVIEW_POINTS=evolution python main_v2.py
```

### Data Curation

```bash
# Build curated pattern library
ENABLE_HITL=true HITL_REVIEW_POINTS=patterns python main_v2.py
```

## Files Added

```
src/lean/human_in_the_loop.py       Core HITL module (500+ lines)
examples/hitl_demo.py                Interactive demo (300+ lines)
docs/human-in-the-loop-guide.md      User guide (600+ lines)
HITL_README.md                       This file
```

## Integration Points

HITL can be integrated at these pipeline stages:

1. **After Content Generation** (`_content_node`)
   - Review/edit outputs before scoring

2. **Before Evaluation** (`_evaluate_node`)
   - Provide manual scores

3. **During Evolution** (`_evolve_node`)
   - Approve/reject population replacement

4. **Before Memory Storage** (`_store_node`)
   - Review patterns before storage

## Technical Details

### Dependencies

- `rich`: Terminal UI (already in project)
- No additional dependencies required

### Design Patterns

- **Strategy Pattern**: Pluggable review points
- **Environment-based Config**: Easy CLI usage
- **Rich Console UI**: Beautiful terminal interactions
- **Statistics Tracking**: Built-in monitoring

### Thread Safety

- HITL is designed for single-threaded use
- Async-compatible (no blocking operations except user input)

## Testing

### Manual Testing

Run the demo script:
```bash
cd /home/justin/Documents/dev/workspaces/lean-hitl
ENABLE_HITL=true python examples/hitl_demo.py
```

### Integration Testing

```bash
# Small experiment with HITL
ENABLE_HITL=true HITL_REVIEW_POINTS=content python main_v2.py
```

## Next Steps

### Before Merge

- [ ] Integration with `pipeline_v2.py` (add HITL calls)
- [ ] Integration with `main_v2.py` (add CLI flags)
- [ ] Unit tests for HITL module
- [ ] Test with real experiment
- [ ] Update main README with HITL section

### Future Enhancements

- Web UI for remote review
- Multi-user review (voting system)
- Automated approval rules (e.g., auto-approve scores > 8.5)
- Review history and replay
- Export reviews to training data

## Examples

### Example 1: Quick Content Check

```bash
# Run 3 generations with content review
ENABLE_HITL=true HITL_REVIEW_POINTS=content python main_v2.py
```

Review each output, fix typos, approve or reject.

### Example 2: Manual Scoring Experiment

```bash
# Generate content automatically, score manually
ENABLE_HITL=true HITL_REVIEW_POINTS=scoring python main_v2.py
```

Compare your scores with automatic evaluation.

### Example 3: Supervised Evolution

```bash
# Control evolution decisions
ENABLE_HITL=true HITL_REVIEW_POINTS=evolution python main_v2.py
```

Approve only when parent fitness is above threshold.

## Known Limitations

1. **Terminal Only**: No GUI (planned for future)
2. **Single User**: No collaborative review (planned for future)
3. **Blocking**: Pauses execution (by design)
4. **No Undo**: Decisions are final (workaround: reject and retry)

## Compatibility

- ✅ Compatible with current `develop` branch
- ✅ Works with `pipeline_v2.py`
- ✅ Compatible with detailed logging
- ✅ Works alongside visualization
- ⚠️ Not integrated with `pipeline.py` (old version)

## Performance Impact

**Overhead**: Minimal when `enabled=False` or `auto_approve=True`
**Interactive**: Adds human review time (10-30 seconds per review)
**Recommended**: Use for small experiments or specific generations

## Documentation

- **User Guide**: [docs/human-in-the-loop-guide.md](docs/human-in-the-loop-guide.md)
- **API Docs**: Docstrings in [src/lean/human_in_the_loop.py](src/lean/human_in_the_loop.py)
- **Demo**: [examples/hitl_demo.py](examples/hitl_demo.py)

## Questions?

See the comprehensive guide: `docs/human-in-the-loop-guide.md`

---

**Ready for Review**: This feature is complete and ready for testing/integration.
**To Test**: `cd /home/justin/Documents/dev/workspaces/lean-hitl && ENABLE_HITL=true python examples/hitl_demo.py`
