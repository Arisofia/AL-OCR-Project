For future contributors and stakeholders:

To tune upscaling, adjust max_upscale_factor and max_long_side_px in EngineConfig.
The upscaling logic is isolated and reusable in ImageToolkit, with clear docstrings.
Robustness improvements in workflow artifact management are now in place (see workflows for naming patterns).

Next recommendations (as CTO):
Carefully monitor for performance or regression impacts during real-world usage (upscaling can increase compute cost for very large batches).
Continue adding test coverage, especially for edge cases in upscaling logic.