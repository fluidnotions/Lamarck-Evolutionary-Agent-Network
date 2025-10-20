# M2 Parallel Development Setup Complete
**Date**: 2025-10-20
**Status**: ✅ Ready for parallel development

---

## Overview

M2 (Step 8: EVOLVE) development is now set up with 4 parallel worktrees, each working on independent features that will be merged into `develop` branch.

---

## Branch Structure

```
master (production)
  ↓
develop (integration branch)
  ├── m2-compaction (worktree: ../lean-compaction)
  ├── m2-selection (worktree: ../lean-selection)
  ├── m2-reproduction (worktree: ../lean-reproduction)
  └── m2-agent-pools (worktree: ../lean-agent-pools)
```

---

## Worktrees Created

| Worktree | Branch | Location | Purpose |
|----------|--------|----------|---------|
| **lean-compaction** | m2-compaction | `/home/justin/Documents/dev/workspaces/lean-compaction` | Forgetting unsuccessful patterns |
| **lean-selection** | m2-selection | `/home/justin/Documents/dev/workspaces/lean-selection` | Parent selection algorithms |
| **lean-reproduction** | m2-reproduction | `/home/justin/Documents/dev/workspaces/lean-reproduction` | Offspring creation with inheritance |
| **lean-agent-pools** | m2-agent-pools | `/home/justin/Documents/dev/workspaces/lean-agent-pools` | Population management |

---

## Development Order

### Phase 1: Parallel Development (4 worktrees)

**Can work simultaneously** - no dependencies:

1. **Compaction** (lean-compaction)
   - Implement 4 strategies: score, frequency, diversity, hybrid
   - Independent of other features
   - Estimated: 1 week

2. **Selection** (lean-selection)
   - Implement 4 strategies: tournament, proportionate, rank, diversity-aware
   - Uses mock AgentPool for testing
   - Estimated: 1 week

3. **Reproduction** (lean-reproduction)
   - Implement asexual and sexual reproduction
   - Uses mock compaction for testing
   - Estimated: 1 week

4. **Agent Pools** (lean-agent-pools)
   - Can use mocks for other features initially
   - Full integration after others merge
   - Estimated: 2 weeks (1 week initial, 1 week integration)

### Phase 2: Sequential Integration

**Merge order** (dependencies):

1. Merge `m2-compaction` → develop (no dependencies)
2. Merge `m2-selection` → develop (no dependencies)
3. Merge `m2-reproduction` → develop (depends on compaction)
4. Merge `m2-agent-pools` → develop (depends on all others)

### Phase 3: Pipeline Integration

After all features merged:
- Update `pipeline_v2.py` with agent pools
- Add `evolve_pools()` method
- Run 20-generation validation

---

## Quick Start Guide

### Working in a Worktree

```bash
# Navigate to worktree
cd /home/justin/Documents/dev/workspaces/lean-compaction

# Check branch
git branch
# * m2-compaction

# Read the README
cat M2_COMPACTION_README.md

# Create files and develop
touch src/lean/compaction.py
# ... implement ...

# Commit and push
git add src/lean/compaction.py
git commit -m "Implement ScoreBasedCompaction"
git push origin m2-compaction

# When done, create PR to develop
# (on GitHub: compare m2-compaction → develop)
```

### Switching Between Worktrees

```bash
# Main repo (develop)
cd /home/justin/Documents/dev/workspaces/hvas-mini

# Compaction worktree
cd /home/justin/Documents/dev/workspaces/lean-compaction

# Selection worktree
cd /home/justin/Documents/dev/workspaces/lean-selection

# Reproduction worktree
cd /home/justin/Documents/dev/workspaces/lean-reproduction

# Agent pools worktree
cd /home/justin/Documents/dev/workspaces/lean-agent-pools

# List all worktrees
cd /home/justin/Documents/dev/workspaces/hvas-mini
git worktree list
```

---

## File Locations by Feature

### Compaction (`lean-compaction`)
- `src/lean/compaction.py`
- `tests/test_compaction.py`
- `examples/compaction_demo.py`

### Selection (`lean-selection`)
- `src/lean/selection.py`
- `tests/test_selection.py`
- `examples/selection_demo.py`

### Reproduction (`lean-reproduction`)
- `src/lean/reproduction.py`
- `tests/test_reproduction.py`
- `examples/reproduction_demo.py`

### Agent Pools (`lean-agent-pools`)
- `src/lean/agent_pool.py`
- `tests/test_agent_pool.py`
- `examples/agent_pool_demo.py`

---

## Testing Strategy

### Per-Feature Tests (in each worktree)
```bash
# Run feature tests
cd /home/justin/Documents/dev/workspaces/lean-compaction
pytest tests/test_compaction.py -v
```

### Integration Tests (after merge to develop)
```bash
# In main repo
cd /home/justin/Documents/dev/workspaces/hvas-mini
git checkout develop
git pull origin develop  # Get latest merges

# Run all M2 tests
pytest tests/test_compaction.py tests/test_selection.py tests/test_reproduction.py tests/test_agent_pool.py -v
```

### System Tests (after pipeline integration)
```bash
# Run 20-generation experiment
python examples/m2_evolution_demo.py
```

---

## Merge Checklist

Before merging feature to develop:

- [ ] All unit tests passing
- [ ] Code documented (docstrings)
- [ ] Demo script works
- [ ] No dependencies on unmerged features (or mocked)
- [ ] PR created and reviewed
- [ ] Conflicts resolved with develop

---

## Configuration

Each feature will use these environment variables:

```bash
# Compaction
COMPACTION_STRATEGY=hybrid
COMPACTION_THRESHOLD=0.5
INHERITED_REASONING_SIZE=100

# Selection
SELECTION_STRATEGY=tournament
TOURNAMENT_SIZE=3
ELITISM_COUNT=1

# Reproduction
REPRODUCTION_STRATEGY=sexual
MUTATION_RATE=0.1
CROSSOVER_RATE=0.5

# Agent Pools
POOL_SIZE=5
EVOLUTION_FREQUENCY=5
```

---

## Communication & Coordination

### If working in parallel:
- Each worktree is independent
- Commit/push frequently to feature branches
- Don't merge to develop until feature complete
- Coordinate merge order (compaction → selection → reproduction → pools)

### If working solo:
- Can work on one feature at a time
- Or switch between worktrees as needed
- Still merge in dependency order

---

## Status Tracking

### Current Status (2025-10-20):

| Feature | Status | Owner | ETA |
|---------|--------|-------|-----|
| Compaction | ⏳ Not Started | - | 1 week |
| Selection | ⏳ Not Started | - | 1 week |
| Reproduction | ⏳ Not Started | - | 1 week |
| Agent Pools | ⏳ Not Started | - | 2 weeks |
| Pipeline Integration | ⏳ Waiting | - | After all merged |

### When Complete:

All features merged → Update status to:
- ✅ Compaction: COMPLETE
- ✅ Selection: COMPLETE
- ✅ Reproduction: COMPLETE
- ✅ Agent Pools: COMPLETE
- 🚀 M2 Evolution: READY

---

## Resources

### Documentation:
- **Master Plan**: `docs/planning/M2_EVOLUTION_PLAN.md`
- **LangGraph Architecture**: `docs/brainstorming/2025-10-20-M2-LANGGRAPH-ARCHITECTURE.md`
- **Worktree READMEs**: In each worktree root
- **Migration Guide**: `docs/MIGRATION_GUIDE.md`
- **Pipeline V2 Guide**: `docs/PIPELINE_V2_GUIDE.md`

### Code:
- **Base develop**: `/home/justin/Documents/dev/workspaces/hvas-mini`
- **BaseAgentV2**: `src/lean/base_agent_v2.py`
- **ReasoningMemory**: `src/lean/reasoning_memory.py`
- **SharedRAG**: `src/lean/shared_rag.py`
- **PipelineV2**: `src/lean/pipeline_v2.py`

### Examples:
- **V2 workflow**: `examples/pipeline_v2_demo.py`
- **Integration**: `examples/v2_workflow_integration.py`

---

## Next Actions

1. ✅ Branches created and pushed
2. ✅ Worktrees set up
3. ✅ READMEs created for each worktree
4. ⏳ Begin implementation (choose a worktree)
5. ⏳ Develop feature
6. ⏳ Test feature
7. ⏳ Merge to develop (in order)
8. ⏳ Integrate with Pipeline V2
9. ⏳ Run 20-generation validation

---

## Git Commands Reference

```bash
# List worktrees
git worktree list

# Remove worktree (when done)
git worktree remove ../lean-compaction

# Prune deleted worktrees
git worktree prune

# Switch to develop in main repo
cd /home/justin/Documents/dev/workspaces/hvas-mini
git checkout develop

# Pull latest develop
git pull origin develop

# Merge feature branch (after PR approved)
git merge m2-compaction
git push origin develop
```

---

## Success Criteria

M2 implementation complete when:
- ✅ All 4 features implemented and tested
- ✅ All features merged to develop
- ✅ Pipeline V2 updated with evolution support
- ✅ 20-generation experiment shows fitness improvement
- ✅ Compaction reduces patterns 100 → 20-30
- ✅ Offspring inherit best patterns
- ✅ Documentation complete

---

**Status**: Setup complete. Ready to begin parallel M2 development!

🚀 **Choose a worktree and start implementing!**
