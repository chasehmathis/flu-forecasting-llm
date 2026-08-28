"""flufc -- an LLM harness that forecasts US influenza hospitalizations.

The pieces, in dependency order:

    config      where the Qwen server is, and what the harness may do
    cache       on-disk HTTP cache, so runs are cheap and reproducible
    llm         transport to the vLLM OpenAI-compatible endpoint
    protocol    Qwen's XML tool-call format: render specs, parse calls
    tools/      the things the model can actually do
    agent       the think -> call -> observe loop
    forecast    quantile forecasts, FluSight-compatible
    evaluate    weighted interval score against the published truth
"""

__version__ = "0.1.0"
