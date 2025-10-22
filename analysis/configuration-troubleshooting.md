# Configuration Troubleshooting Guide

**Created**: 2025-10-22
**Status**: Active

## Common Configuration Issues

### Issue 1: Visualization Crash - AgentPool Method Error

**Symptom**:
```
AttributeError: 'AgentPool' object has no attribute 'select_active_agent'
```

**Cause**: `HierarchicalVisualizer` was calling wrong method name.

**Fix**: Changed `pool.select_active_agent()` to `pool.select_agent(strategy="best")` in `src/lean/visualization.py:458`

**Status**: ✅ Fixed in commit 8d9e244

---

### Issue 2: LLM Provider/Model Mismatch

**Symptom**:
```
openai.NotFoundError: Error code: 404 - {'error': {'message': 'The model `claude-3-haiku-20240307` does not exist or you do not have access to it.'
```

**Cause**: Your `.env` has:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=...
```

But the code is trying to use Claude models through OpenAI's API, which doesn't work.

**Solution Options**:

#### Option 1: Use Anthropic API with Claude Models (Recommended)

Update `.env`:
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here
# Comment out or remove:
# OPENAI_API_KEY=...
```

This will use Claude models (`claude-3-haiku-20240307`, `claude-3-5-sonnet-20241022`, etc.) through Anthropic's API.

#### Option 2: Use OpenAI Models with OpenAI API

Update `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
```

Then update model names in the codebase to OpenAI models:
- `gpt-4o` (latest GPT-4 with optimized performance)
- `gpt-4o-mini` (faster, cheaper GPT-4)
- `gpt-4-turbo` (previous generation)
- `gpt-3.5-turbo` (fastest, cheapest)

**Where to change models**: The model is set when creating the LLM in `src/lean/base_agent.py` and `src/lean/coordinator.py`.

---

### Issue 3: Visualization Not Showing

**Symptom**: No visualization appears in terminal when running experiments.

**Check**:
1. Is visualization enabled in config?
   ```yaml
   # config/experiments/your_config.yml
   visualization:
     enabled: true
   ```

2. Are you using the correct terminal? Visualization requires:
   - A terminal that supports Rich library rendering
   - Direct terminal execution (not through log files)

**How to View**:
```bash
# Run directly in terminal (visualization shows in real-time):
./run_experiment.sh fast_test

# Or use uv directly:
uv run python main.py --config fast_test
```

**Note**: Visualization appears IN THE TERMINAL where you run the command, NOT in log files.

---

### Issue 4: Empty Experiment Logs

**Symptom**: All 20 generations fail immediately, no content generated.

**Check**:
1. Is the LLM provider/key configured correctly? (See Issue 2)
2. Are there any API errors in the logs?
3. Is network connectivity working?

**Diagnostic Command**:
```bash
# Test LLM connection
uv run python -c "
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model='claude-3-haiku-20240307')
response = llm.invoke('Say hello')
print(response.content)
"
```

---

## Configuration Files Quick Reference

### `.env` (API Keys)
```bash
# For Anthropic/Claude:
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI/GPT:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Optional:
TAVILY_API_KEY=tvly-...  # For research feature
```

### `config/experiments/` (Experiment Settings)
```yaml
experiment:
  name: "My Experiment"
  population_size: 3
  evolution_frequency: 5
  total_generations: 20

# Enable/disable features:
research:
  enabled: true  # Requires TAVILY_API_KEY

visualization:
  enabled: true  # Shows in terminal

hitl:
  enabled: false  # Not implemented yet
```

### Model Configuration
Models are hardcoded in:
- `src/lean/base_agent.py` - Content agents
- `src/lean/coordinator.py` - Coordinator agent
- `src/lean/specialists.py` - Specialist agents

Search for `ChatAnthropic` or `ChatOpenAI` to change models.

---

## Testing After Configuration Changes

### Quick Test (3 generations, no research)
```bash
./run_experiment.sh fast_test
```

### Full Test (20 generations, with research)
```bash
./run_experiment.sh default
```

### Check Logs
```bash
# Latest log:
ls -lt experiment_logs/ | head -n 1

# View log:
less experiment_logs/20251022_HHMMSS_config.log
```

---

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Hierarchical Visualization** | ✅ Fixed | Commit 8d9e244 |
| **Experiment Logging** | ✅ Working | Logs to `experiment_logs/` |
| **Automatic Scoring** | ✅ Working | See `analysis/scoring-system-explained.md` |
| **YAML Configuration** | ✅ Working | Prompts and experiments |
| **LLM Provider** | ⚠️ User Config | Need correct API key in `.env` |
| **HITL** | ❌ Not Implemented | Config exists but not wired up |

---

## Next Steps for Working System

1. **Fix your `.env`**:
   ```bash
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=your_actual_anthropic_key
   ```

2. **Test visualization**:
   ```bash
   ./run_experiment.sh fast_test
   ```

   You should see the hierarchical visualization in your terminal showing:
   - L1: Coordinator
   - L2: Content Agents (intro/body/conclusion) in pools
   - L3: Specialist Agents
   - Pool evolution stats
   - Coordinator workflow phases

3. **If visualization works, try full experiment**:
   ```bash
   ./run_experiment.sh default
   ```

---

**Last Updated**: 2025-10-22 after commit 8d9e244
