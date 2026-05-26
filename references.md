# DaCapo Contribution References

This file tracks bugs, issues, and solutions encountered and resolved in this project.

---

## Entry 1 — 2026-05-14

### Issue Description
`datasplit_generator.py` (containing the widely-used `DataSplitGenerator` class) was accidentally deleted in commit `f5c26fdb` ("avoid prints, replace with logger.info"). The commit was intended to replace `print()` calls with `logger.info()`, but instead deleted the entire 1073-line file. This broke all examples that import `DataSplitGenerator` from `dacapo.experiments.datasplits`.

Additionally, `DataSplitGenerator` was never re-exported from the package `__init__.py`, so even if the file existed users could not import it via the documented API.

**Related GitHub issues:** [#296 — Non-integer inputs in DataSplitGenerator](https://github.com/janelia-cellmap/dacapo/issues/296)

### Root Cause Analysis
The commit `f5c26fdb` (Feb 17, 2025) deleted `dacapo/experiments/datasplits/datasplit_generator.py` while it only needed to replace one `print()` call on line 88 with `logger.info()`. This appears to be an accidental deletion.

The file also had a circular-import risk: it imported `TrainValidateDataSplitConfig` from `dacapo.experiments.datasplits` (the package), while now the package `__init__.py` would import from it.

### Solution Applied
1. Restored `datasplit_generator.py` from git history (pre-deletion state).
2. Replaced the lone `print()` call in `resize_if_needed()` with `logger.info()`.
3. Fixed the circular import by importing `TrainValidateDataSplitConfig` directly from its module file.
4. Added `DataSplitGenerator` to `dacapo/experiments/datasplits/__init__.py`.
5. Added a `_validate_resolution()` helper that warns when non-integer voxel resolutions are passed (addresses issue #296).

### Changes Made
| File | Change |
|------|--------|
| `dacapo/experiments/datasplits/datasplit_generator.py` | Restored; `print()` → `logger.info()`; fixed circular import; added non-integer resolution warning |
| `dacapo/experiments/datasplits/__init__.py` | Added `from .datasplit_generator import DataSplitGenerator` |

### Prevention Notes
- When making style-only commits (print → logger), run `git diff --stat` before committing to verify no files are accidentally deleted.

---

## Entry 2 — 2026-05-14

### Issue Description
Two `logger.info()` calls had variables passed as separate positional arguments (mimicking `print()` syntax). Python's logging module silently ignores extra args when the message has no `%`-format specifiers, so the variable values were never logged.

1. `dacapo/predict.py:150` — worker file path silently dropped.
2. `dacapo/experiments/run_config.py:498-501` — saved BioImage.IO package path silently dropped.

Both introduced in commit `f5c26fdb`.

### Solution Applied
Changed both to f-strings. For `run_config.py`, the `save_bioimageio_package()` return value is captured in a variable first so the side effect is explicit.

### Changes Made
| File | Change |
|------|--------|
| `dacapo/predict.py` | `logger.info("...", worker_file)` → `logger.info(f"...{worker_file}")` |
| `dacapo/experiments/run_config.py` | Split into `saved_path = save_bioimageio_package(...)` + `logger.info(f"package path: {saved_path}")` |

### Prevention Notes
Always use f-strings when logging variables. Never pass extra positional args to `logger.*`.

---

## Entry 3 — 2026-05-14

### Issue Description
`HotDistancePredictor` had two hardcoded magic numbers — `epsilon = 5e-2` and `threshold = 0.8` — marked with `# TODO: should be a config parameter`. Users had no way to tune them without modifying source code.

### Solution Applied
Added `epsilon` and `threshold` as optional `attr.ib` fields to `HotDistanceTaskConfig` with the existing defaults (fully backward-compatible), wired through `HotDistanceTask` to `HotDistancePredictor`.

### Changes Made
| File | Change |
|------|--------|
| `dacapo/experiments/tasks/hot_distance_task_config.py` | Added `epsilon` and `threshold` attr fields |
| `dacapo/experiments/tasks/hot_distance_task.py` | Forwarded both values to `HotDistancePredictor` |
| `dacapo/experiments/tasks/predictors/hot_distance_predictor.py` | Replaced hardcoded values with constructor params |

### Prevention Notes
Avoid hardcoding tunable hyperparameters in predictor classes. Always expose them through the config from the start.
