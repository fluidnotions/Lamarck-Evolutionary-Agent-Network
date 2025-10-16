# The NumPy Serialization Problem

## The Issue

**Error**: `TypeError: Type is not msgpack serializable: numpy.float64`

**When**: During LangGraph workflow execution, when the system tries to checkpoint state to memory.

**Why It's Tricky**: The error appears inconsistently and can be hard to trace because numpy types look like regular Python floats/ints until serialization happens.

---

## Root Cause

LangGraph uses msgpack to serialize state for checkpointing. Msgpack doesn't natively support numpy types:
- `numpy.float64` ≠ Python `float`
- `numpy.int64` ≠ Python `int`
- `numpy.ndarray` ≠ Python `list`

Even though these look identical when printed or used in calculations, they're different types under the hood.

---

## Where NumPy Types Sneak In

In this codebase, numpy is used in several places:

### 1. **Trust Manager** (`weighting/trust_manager.py`)
```python
import numpy as np

def compute_performance_signal(intro_score, body_score):
    signal = np.mean([intro_score, body_score])  # Returns np.float64!
    return signal
```

### 2. **Semantic Distance** (`hierarchy/semantic.py`)
```python
import numpy as np

def compute_semantic_distance(vec_a, vec_b):
    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)  # np.float64
    distance = (1.0 - similarity) / 2.0  # Still np.float64!
    return distance
```

### 3. **Memory Decay** (`memory/decay.py`)
```python
import numpy as np

def calculate_decay_factor(age_seconds):
    return np.exp(-self.decay_lambda * age_seconds)  # np.float64
```

### 4. **Agent Confidence Calculations**
Any calculations involving numpy operations produce numpy types:
```python
weights = [o["confidence"] / total_confidence for o in outputs]
# If total_confidence came from numpy, ALL weights are np.float64
```

---

## Why Our First Fix Wasn't Enough

We added `_sanitize_state()` in `pipeline.py`, but we only called it in:
1. `_evaluate_wrapper()` - for scores
2. `_evolution_node()` - at the end

**Problem**: State gets modified in many other places:
- Agents update state directly in `__call__()`
- Async coordinator adds timing data
- Weight updates happen throughout execution
- Any intermediate node can introduce numpy types

The sanitization happens too late - after numpy types are already in state and LangGraph tries to checkpoint them.

---

## The Real Solution: Sanitize All State Updates

We need to intercept **every** state modification, not just at specific nodes.

### Option 1: Wrapper for Every Node
Wrap every node function to sanitize its return:
```python
def _wrap_node(node_func):
    def wrapper(state):
        result = node_func(state)
        return _sanitize_state(result)
    return wrapper

workflow.add_node("intro", _wrap_node(self.agents["intro"]))
```

### Option 2: Custom StateGraph with Auto-Sanitization
Subclass StateGraph to automatically sanitize after every node:
```python
class SanitizedStateGraph(StateGraph):
    def add_node(self, name, func):
        wrapped = lambda state: _sanitize_state(func(state))
        super().add_node(name, wrapped)
```

### Option 3: Fix at the Source
Convert numpy types to Python types immediately when created:
```python
# In trust_manager.py
signal = float(np.mean([intro_score, body_score]))  # Explicit cast

# In semantic.py
distance = float((1.0 - similarity) / 2.0)  # Explicit cast

# In all calculations
return float(result)  # Always cast before returning
```

**Option 3 is the most robust** because it prevents numpy types from ever entering state in the first place.

---

## Debugging Tips

### 1. Find the Numpy Type
Add this helper to find where numpy types are hiding:
```python
def find_numpy_types(obj, path=""):
    """Recursively find numpy types in nested structures."""
    import numpy as np

    if isinstance(obj, (np.integer, np.floating, np.ndarray)):
        print(f"Found {type(obj)} at: {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_numpy_types(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            find_numpy_types(item, f"{path}[{i}]")

# Use before checkpointing
find_numpy_types(state, "state")
```

### 2. Check After Each Node
Add logging to see which node introduces numpy:
```python
async for event in self.app.astream(state, config):
    find_numpy_types(event, "after_node")
    yield event
```

### 3. Test Serialization Explicitly
```python
import ormsgpack

try:
    ormsgpack.packb(state)
    print("State is serializable")
except TypeError as e:
    print(f"Serialization failed: {e}")
    find_numpy_types(state, "state")
```

---

## Long-Term Fix (Recommended)

1. **Add type annotations** to all functions that return numeric values
2. **Always cast numpy results** to Python types at calculation boundaries
3. **Add a pre-commit hook** that checks for numpy imports in state-modifying code
4. **Create a linter rule** that flags `return np.xxx()` without explicit casting

Example pattern:
```python
def calculate_something() -> float:  # Type hint forces awareness
    result = np.mean(values)
    return float(result)  # Explicit cast ensures Python type
```

---

## Why This Matters

This isn't just a technical nuisance - it reveals a design principle:

**State should be serialization-agnostic**

If your state can't be pickled, msgpacked, or JSON-dumped, you're coupling your logic to your compute runtime. This makes:
- Checkpointing fragile
- Distributed execution impossible
- Debugging harder (can't easily inspect state)
- Testing more complex (can't mock state easily)

Keep state clean: use Python native types wherever possible. Use numpy for calculations, but cast results before they touch state.

---

## Current Status

We've added sanitization in `pipeline.py`, but it's not comprehensive enough. Numpy types are still slipping through from:
- Trust weight calculations
- Semantic distance computations
- Confidence score aggregations

**Next Steps**:
1. Add explicit `float()` casts in all modules that use numpy
2. Wrap all node functions with sanitization
3. Add tests that verify state serializability after each node

This is a known issue we're actively working on fixing.
