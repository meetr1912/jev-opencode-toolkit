# jev-opencode-toolkit benchmark results

> **OFFLINE / DETERMINISTIC: no TYPESAFE_API_KEY, no network. Every number below comes from the jev_toolkit offline stub through the strict-validation ask path.**

- Command: `uv run --no-sync python benchmarks/bench.py --offline --out results`
- Engine: offline stub, deterministic, no RNG
- Python: 3.12.14

## Commands

```bash
uv run --no-sync python benchmarks/bench.py --offline --out results
uv run --no-sync jev-lock check --lock calibration/decisions.lock.json \
  --questions examples/pinned_questions.json --samples calibration/samples.json
```

## SPEED - one batched request vs N single-question requests

offline stub (simulated 20 ms round-trip). N = 80, simulated round-trip 0.02 s.

| strategy | requests | total s | ms/question |
| --- | ---: | ---: | ---: |
| 80 separate asks | 80 | 1.60 | 20.0 |
| one batched ask | 1 | 0.02 | 0.2 |

**Timing ratio:** ~78.2x fewer wall-seconds batched (raw best-of; display seconds are snapped for stable CI output). **Call-count ratio (exact):** 80:1.

## PARALLELISM - bounded concurrency over batches

offline stub (simulated 50 ms round-trip per group). 80 questions -> 8 groups, concurrency 1 vs 8.

| concurrency | wall s | requests |
| ---: | ---: | ---: |
| 1 | 0.40 | 8 |
| 8 | 0.05 | 8 |

**Speedup:** ~7.4x (raw best-of). **Order preserved:** True.

## ACCURACY - oracle vs miscalibrated through the same validation path

offline stub (simulated 0 ms; exact oracle vs overconfident miscalibrated). 60 events (20 noul + 40 categorical).

| mode | Brier | log loss | ECE | multiclass Brier | agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle (reference) | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| miscalibrated | 0.101250 | 0.298919 | 0.225000 | 0.013763 | 1.000000 |

Oracle matches the reference (Brier=0, ECE=0, multiclass Brier=0, agreement=1.0): **True**. Miscalibrated is strictly worse on every metric: **True**.

Teeth: `mode="corrupt"` raises `JevError` -> **True** (`noul_00: noul answer missing or not a probability in [0, 1]`).

## CALIBRATION - committed samples + lock

committed samples + lock, verified by the jev-lock CLI.

| metric | value |
| --- | ---: |
| brier | 0.105578 |
| log_loss | 0.369756 |
| ece | 0.214444 |
| accuracy | 0.583333 |
| n | 36 |

| qid | threshold | coverage | accuracy | n |
| --- | ---: | ---: | ---: | ---: |
| amount_confidence | 0.60 | 0.5833 | 1.0000 | 12 |
| has_signature | 0.70 | 0.5000 | 1.0000 | 12 |
| is_invoice | 0.65 | 0.5833 | 1.0000 | 12 |

`questions_sha256`: `b92c7ad2e403a087a9a847cbb8b6b9a792f3f71fd6017824d7c6c634efa729d3`

CLI check: `OK: calibration lock verified` (exit 0).

### Teeth - tampered copies fail

| tamper | exit code | CLI stderr |
| --- | ---: | --- |
| metric (`brier += 0.5`) | 1 | FAIL: metric drift: brier lock=0.605578 actual=0.10557777777777778 |
| question (instruction edited) | 1 | FAIL: questions hash drift: lock=b92c7ad2e403a087a9a847cbb8b6b9a792f3f71fd6017824d7c6c634efa729d3 file=20418628602660bd58e39277a94126b02cccc9f1fca15165bc77cb8b82f94a95 |

## Headline

- Speed timing ratio: **~78.2x** (raw best-of); call-count ratio **80:1** (exact).
- Parallel speedup: **~7.4x** (concurrency 8, raw best-of).
- Oracle vs miscalibrated Brier: **0.000000 -> 0.101250**; ECE: **0.000000 -> 0.225000**.
- Calibration lock check exit code: **0**.

## Reproduce

```bash
# 1. regenerate samples, lock, results (samples/lock deterministic; timings measured)
uv run --no-sync python benchmarks/bench.py --offline --out results

# 2. verify the committed lock against the pinned rubric and samples
uv run --no-sync jev-lock check --lock calibration/decisions.lock.json \
  --questions examples/pinned_questions.json --samples calibration/samples.json
```
