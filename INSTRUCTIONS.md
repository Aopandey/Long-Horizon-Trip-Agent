# Running the agent on your computer

These steps are for a teammate who runs the model locally for **dev tuning**. Results go straight into the team's Supabase database, so everyone can see them. Message Youssef if anything here does not work.

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
- From Youssef, sent privately: the Supabase **project URL** and **your own secret key** (`sb_secret_...`).

## 1. Install the model

Install Ollama, start it, then pull the model. It is a 7.6 GB download.

```bash
ollama pull qwen3.5:9b
```

## 2. Get the code

The repo has the code and the 80 scenarios. Your runs read the scenarios from Supabase, which holds the same data.

```bash
git clone https://github.com/Aopandey/Long-Horizon-Trip-Agent.git
cd Long-Horizon-Trip-Agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,supabase]"
```

If you already cloned the older `feat/core-implementation` branch, switch to `main` with `git switch main && git pull`. Nothing else changes.

## 3. Add your keys

Copy the template and open it:

```bash
cp .env.example .env
```

Fill in these two lines. No quotes, no spaces around `=`:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_KEY=sb_secret_...
```

- `.env` is ignored by git, so it is never committed. Keep it that way: never paste the key into a chat, an issue, or a commit.
- Your key can read, change and **delete** everything in the team database. Only run the commands on this page.
- Run every `trip-agent` command from the repo folder, because that is where `.env` is read from.

## 4. Check that everything works

Run these in order. Each one should end without an error.

```bash
pytest -m "not ollama" -q                 # all tests pass: the code works without the model
trip-agent validate --backend supabase    # dataset OK, 80 scenarios, hash f3270941decf0d9a
trip-agent check-llm                      # OK: Ollama is running and the model supports tools
```

The hash must be exactly `f3270941decf0d9a`. If it is different, run `git pull` and try again, then tell Youssef.

Then run one short live episode. It reads the scenario from Supabase, prints each tool call as it happens and a score at the end, and stores nothing. Expect about 5 to 15 minutes.

```bash
trip-agent smoke --backend supabase --scenario dev-h05-001
```

While it runs, check in a second terminal that the model is on the GPU. The `PROCESSOR` column should say `100% GPU`.

```bash
ollama ps
```

If `check-llm` or `smoke` fails, or `ollama ps` shows CPU, stop and send Youssef the error. This model ships in a newer weight format (NVFP4) that some GPUs may not run well.

## 5. Run a small dev batch

Create the file `configs/dev-archer.yaml` with this content. (Other teammates: use your own name instead of `archer`, in the file name and in `name:`.)

```yaml
# Dev tuning: data from Supabase, results saved to Supabase.
name: dev-archer-v1        # must be unique in the team database: put your name in it
split: dev
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
storage: {backend: supabase, dataset: main}
```

Run the first 6 runs: scenario `dev-h05-001` in both conditions with all three variants.

```bash
trip-agent run configs/dev-archer.yaml --limit 6
```

Each finished run is saved to Supabase immediately with its score. Then print the summary table:

```bash
trip-agent report configs/dev-archer.yaml
```

The whole config is 60 runs (10 scenarios x 2 conditions x 3 variants). Runs are **resumable**: running the same command again skips finished runs, so you can stop with Ctrl+C and continue later, even from another computer with the same config. Leave out `--limit` to run everything that is left.

Keep the computer awake during long batches. On a Mac, prefix the command with `caffeinate -i`; on Linux, run it inside `tmux`.

## 6. Results

Nothing to send: the runs are already in the team database. In the Supabase dashboard, the **`run_results`** view has one row per run with its scores. Filter by `experiment` to see yours.

Do tell Youssef your **GPU model** and **how long the 6 runs took**. We need this to plan the full 720 runs.

## Rules that keep the results usable

- **Do not run `configs/eval.yaml`.** The eval set runs once, after the settings are frozen. It refuses to start without `--allow-eval`; please do not add that flag.
- **One experiment name = one fixed set of settings.** To try something (for example `think: true` or a bigger budget), copy your config, give it a new name such as `dev-archer-v2-think`, and tell the team what you changed. Never reuse someone else's experiment name.
- **Do not change code or prompts on your own.** Suggest changes to the team instead, so every run stays comparable.
- **Do not delete or edit rows in Supabase.** If something looks wrong, tell Youssef.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `trip-agent: command not found` | Activate the venv: `source .venv/bin/activate`. |
| `No module named 'trip_agent'` | Run `python -m trip_agent ...` instead of `trip-agent ...`, and make sure the repo is not in a synced folder. |
| Commands hang with no output | The repo is probably in a synced folder (iCloud, OneDrive). Clone it again somewhere else. |
| `set SUPABASE_URL and SUPABASE_SERVICE_KEY` | `.env` is missing, still has placeholders, or you are not in the repo folder. |
| `dataset 'main' not found in Supabase` | Either `SUPABASE_URL` is not the team project, or you used the publishable key. You need the **secret** key (`sb_secret_...`); the publishable key sees no rows. |
| `violates row-level security policy` | Same cause: use the secret key, not the publishable key. |
| `Connection refused` on port 11434 | Ollama is not running. Start the Ollama app, or run `ollama serve`. |
| `model 'qwen3.5:9b' is not available` | Run `ollama pull qwen3.5:9b`. |
| A run shows `error` in the progress line | Run the same command again; failed runs are retried. Tell Youssef the run key if it keeps failing. |
| Runs are extremely slow | Check `ollama ps`. If it does not say `100% GPU`, the model is running on the CPU. |
