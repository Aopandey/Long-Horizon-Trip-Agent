# Running the agent on your computer

These steps are for a teammate who runs the model locally for **dev tuning**. Message Youssef if anything here does not work.

## What we are doing

We are **not training the model**. Qwen3.5 9B's weights never change. Your computer only *runs* the model (inference), the same way ChatGPT runs when you use it.

The project tunes the **agent around the model**: its prompts, verification settings, repair limit and tool-call budget. Then it tests whether verification and recovery help when search results contain false facts.

| Phase | Scenarios | What happens |
| --- | --- | --- |
| Dev tuning (now) | 40 `dev` | Run, read the scores, adjust settings, run again |
| Freeze | none | Settings are locked and tagged |
| Final test (later, once) | 40 `eval` | The 720 runs for the report. **Do not run these now.** |

## What you need

- A GPU with at least 10-12 GB of memory (NVIDIA), or an Apple Silicon Mac with 16 GB or more. The model used 9 GB on a Mac.
- About 10 GB of free disk space.
- Python 3.11 or newer, git, and [Ollama](https://ollama.com/download).
- A folder that is **not** synced by iCloud, OneDrive or Dropbox. Synced folders can evict files and make Python hang. On a Mac, avoid Desktop and Documents; use something like `~/Projects`.

## 1. Install the model

Install Ollama, start it, then pull the model. It is a 7.6 GB download.

```bash
ollama pull qwen3.5:9b
```

## 2. Get the code and the data

Until the pull requests are merged, the `synthetic-data` branch has both the code and the 80 scenarios.

```bash
git clone -b synthetic-data https://github.com/Aopandey/Long-Horizon-Trip-Agent.git
cd Long-Horizon-Trip-Agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 3. Check that everything works

Run these in order. Each one should end without an error.

```bash
pytest -m "not ollama" -q          # 50 passed: the code works without the model
trip-agent validate data           # dataset OK: 80 scenarios
trip-agent check-llm               # OK: Ollama is running and the model supports tools
```

Then run one short live episode. It prints each tool call as it happens and a score at the end. Expect about 5 to 15 minutes.

```bash
trip-agent smoke --data-dir data --scenario dev-h05-001
```

While it runs, check in a second terminal that the model is on the GPU. The `PROCESSOR` column should say `100% GPU`.

```bash
ollama ps
```

If `check-llm` or `smoke` fails, or `ollama ps` shows CPU, stop and send Youssef the error. This model ships in a newer weight format (NVFP4) that some GPUs may not run well.

## 4. Run a small dev batch

Create a file `configs/dev-local.yaml` with this content:

```yaml
# Dev tuning on one machine, results stored locally (no Supabase needed).
name: dev-local-v1
split: dev
data_dir: data
horizons: [5]            # start with the 1-day scenarios
variants: [standard, verification, recovery]
conditions: [clean, corrupted]
repeats: 1
corruption: {rate: 0.2, seed: 0}
budget: {base: 20, per_commitment: 5}
llm:
  model: qwen3.5:9b
  temperature: 0.6
  num_ctx: 32768
  think: false
max_turns: 200
concurrency: 1
storage: {backend: local, root: runs_local}
```

Run the first 6 runs: scenario `dev-h05-001` in both conditions with all three variants.

```bash
trip-agent run configs/dev-local.yaml --limit 6
```

Then summarize them:

```bash
trip-agent report configs/dev-local.yaml
```

The whole config is 60 runs (10 scenarios x 2 conditions x 3 variants). Runs are **resumable**: running the same command again skips finished runs, so you can stop with Ctrl+C and continue later. Leave out `--limit` to run everything that is left.

Keep the computer awake during long batches. On a Mac, prefix the command with `caffeinate -i`; on Linux, run it inside `tmux`.

## 5. Send the results back

Zip these two folders and send them to Youssef:

```bash
zip -r dev-local-v1-results.zip reports/dev-local-v1 runs_local/experiments/dev-local-v1
```

`reports/` has the summary tables (`summary.csv`, `runs.csv`, `comparisons.json`). `runs_local/` has every run's full log, which we need to see why a run failed.

## Rules that keep the results usable

- **Do not run `configs/eval.yaml`.** The eval set runs once, after the settings are frozen. It refuses to start without `--allow-eval`; please do not add that flag.
- **Do not change code or prompts on your own.** If you want to try a setting (for example `think: true` or a bigger budget), copy the config, give it a new `name` such as `dev-local-v2-think`, and tell the team what you changed. Every experiment name must mean one fixed set of settings.
- **Note the hardware:** tell us the GPU model and how long the 6 runs took. We need this to plan the full 720 runs.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `trip-agent: command not found` | Activate the venv: `source .venv/bin/activate`. |
| `No module named 'trip_agent'` | Run `python -m trip_agent ...` instead of `trip-agent ...`, and make sure the repo is not in a synced folder. |
| Commands hang with no output | The repo is probably in a synced folder (iCloud, OneDrive). Clone it again somewhere else. |
| `Connection refused` on port 11434 | Ollama is not running. Start the Ollama app, or run `ollama serve`. |
| `model 'qwen3.5:9b' is not available` | Run `ollama pull qwen3.5:9b`. |
| A run shows `error` in the progress line | Run the same command again; failed runs are retried. Send us the log if it keeps failing. |
| Runs are extremely slow | Check `ollama ps`. If it does not say `100% GPU`, the model is running on the CPU. |
