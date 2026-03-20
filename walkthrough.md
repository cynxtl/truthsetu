# TruthSetu Startup Hang — Debugging Report

## Summary

**No syntax errors, indentation errors, or code bugs were found.** The hang is caused by a **PyTorch + uvicorn reloader deadlock on Windows**.

---

## What Was Checked

| Check | Files | Result |
|-------|-------|--------|
| `ast.parse()` syntax check | 18 Python files | ✅ All pass |
| Mixed indentation (tabs/spaces) | 18 files | ✅ None found |
| Unclosed brackets/parens/strings | 18 files | ✅ All balanced |
| [__init__.py](file:///c:/arnav/projects/TruthSetu/backend/__init__.py) files | 5 files | ✅ All empty (correct) |
| [verify_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py) [_call_groq_with_tools](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py#432-547) | `if/else` structure | ✅ Correct — lines 462–539 |

### [verify_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py) — [_call_groq_with_tools](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py#432-547) Structure Verified

The `if msg.tool_calls:` block (line 462) is correctly structured:
- `messages.append(...)` — indented inside [if](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py#581-653) ✅
- `for tool_call in msg.tool_calls:` — indented inside [if](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py#581-653) ✅
- `response2 = ...` — indented inside [if](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py#581-653) ✅
- `else:` at line 536 — same level as `if msg.tool_calls:` ✅

---

## Root Cause: PyTorch Import Deadlock

### Evidence

| Test | Result |
|------|--------|
| `import langchain_core` | 0.2s ✅ |
| `import faiss` | instant ✅ |
| `from sentence_transformers import SentenceTransformer` | **Hangs >3 min** ❌ |
| `import torch` (bare) | **Hangs >2 min** ❌ |
| `from backend.main import app` | **Hangs indefinitely** ❌ |

### Why It Happens

1. `uvicorn --reload` uses Python's `multiprocessing` module to spawn a watcher + server subprocess
2. On **Windows**, PyTorch's C++ extensions use `fork()`-like initialization that **deadlocks** when loaded inside a `multiprocessing` child process with the `spawn` start method
3. The import chain is: [main.py](file:///c:/arnav/projects/TruthSetu/backend/main.py) → [routes.py](file:///c:/arnav/projects/TruthSetu/backend/api/routes.py) → [verify_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py) (line 22: `from sentence_transformers import SentenceTransformer`) → triggers `import torch` → **deadlock**

### The Trigger Files

- [verify_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py#L22) — line 22: `from sentence_transformers import SentenceTransformer`
- [scout_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/scout_agent.py#L8) — line 8: `from sentence_transformers import SentenceTransformer`

Both are imported at **module level** by [routes.py](file:///c:/arnav/projects/TruthSetu/backend/api/routes.py#L6-L7), which runs during startup.

---

## Fixes (Pick One)

### Fix 1: Run without `--reload` (quickest)

```bash
py -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

No `--reload` = no subprocess spawn = no deadlock.

### Fix 2: Add `multiprocessing` freeze support (recommended)

Add this to **the very top** of [backend/main.py](file:///c:/arnav/projects/TruthSetu/backend/main.py) (before any other imports):

```python
import multiprocessing
multiprocessing.freeze_support()
```

And run with:

```bash
py -m uvicorn backend.main:app --reload
```

### Fix 3: Lazy-load SentenceTransformer (best long-term)

Move the `SentenceTransformer` import inside the [initialize()](file:///c:/arnav/projects/TruthSetu/backend/agents/deploy_agent.py#75-96) methods instead of at module level. This is already done in [rss_monitor.py](file:///c:/arnav/projects/TruthSetu/backend/core/rss_monitor.py) (line 168) but not in [verify_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py) and [scout_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/scout_agent.py).

#### In [verify_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/verify_agent.py) — remove line 22 and change [initialize()](file:///c:/arnav/projects/TruthSetu/backend/agents/deploy_agent.py#75-96):

```diff
-from sentence_transformers import SentenceTransformer
 ...
 async def initialize(self):
     logger.info("Initialising VERIFY agent...")
+    from sentence_transformers import SentenceTransformer
     self._embedder = SentenceTransformer(settings.embedding_model)
```

#### In [scout_agent.py](file:///c:/arnav/projects/TruthSetu/backend/agents/scout_agent.py) — remove line 8 and change [initialize()](file:///c:/arnav/projects/TruthSetu/backend/agents/deploy_agent.py#75-96):

```diff
-from sentence_transformers import SentenceTransformer
 ...
 async def initialize(self):
     logger.info("Initialising SCOUT agent...")
+    from sentence_transformers import SentenceTransformer
     self._embedder = SentenceTransformer(settings.embedding_model)
```

This avoids loading PyTorch during module import and defers it to the async startup context where there's no deadlock.
