# Evaluating LLMs' ability to predict their success on tasks

This repository contains code to reproduce experiments from the paper "Do LLMs Know What They Are Capable Of?"

Experiments are implemented using the [Inspect framework](https://inspect.aisi.org.uk/).

- **Experiment 1**: Elicit in-advance confidence estimates on BigCodeBench (BCB) tasks.
- **Experiment 2**: Evaluate LLM profitability on a sequence of work contracts, where the LLM decides whether to accept or decline each contract.
- **Experiment 3**: Elicit confidence estimates step-by-step through multi-step SWE Bench tasks.

## Requirements and Installation

Requires Python (tested with Python 3.13), Docker, and API access for LLM inference. Other dependencies are listed in pyproject.toml.

Install with uv:
```bash
uv sync
uv pip install -e .
```

Set your API key as described in the [Inspect documentation](https://inspect.aisi.org.uk/providers.html). You can either set an API key as an environment variable, e.g.
```bash
export OPENAI_API_KEY=sk-...
```
or create a `.env` file in the root directory containing, e.g. `OPENAI_API_KEY=sk-...`.

## Experiment 1

- Script to run the BCB benchmark: `src/do_llms_know_what_theyre_capable_of/experiment1/bcb_benchmark.py`
  - Docker must be running when the script is called.
- Script to elicit confidence estimates: `src/do_llms_know_what_theyre_capable_of/experiment1/bcb_predict.py`.
- Script to analyze results: `src/do_llms_know_what_theyre_capable_of/experiment1/analysis.py`

## Experiment 2

- Before running Experiment 2, benchmark results (`.eval` files) from `src/do_llms_know_what_theyre_capable_of/experiment1/bcb_benchmark.py` must be obtained and placed in 
`src/do_llms_know_what_theyre_capable_of/experiment2/bcb_benchmark_logs/`.
  - An example log (`gpt-41-mini-bcb_benchmark.eval`) is included for reference. To use this example, set `model="openai/gpt-4.1-mini-2025-04-14"` in `sequence_of_contracts.py`.
- Script to run the experiment: `src/do_llms_know_what_theyre_capable_of/experiment2/sequence_of_contracts.py`.
  - The model string you pass must exactly match `eval.model` recorded in the benchmark log `.eval` files.
- Script to analyze results: `src/do_llms_know_what_theyre_capable_of/experiment2/analysis.py`

## Experiment 3

- Script to run the experiment: `src/do_llms_know_what_theyre_capable_of/experiment2/swebench_predict_each_step.py`.
  - Docker must be running when the script is called.
- Script to analyze results: `src/do_llms_know_what_theyre_capable_of/experiment3/analysis.py`
