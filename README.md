# jev-opencode-toolkit

A lean TypeSafe Jev toolkit for opencode. One `ask()` sends many independent typed questions over a single state in **one** HTTP request, validates every answer strictly, and returns typed results — plus bounded concurrency, a content-hash cache, pinned JSON rubrics, a calibration lock, and a 3-tool stdio MCP server. MIT.

## Why

Sending N questions in one body is roughly an order of magnitude cheaper and faster than N separate calls (~12x cheaper / ~10x faster in the TypeSafe parallel-questions cookbook). This toolkit defaults to exactly that: `ask(state, questions)` → one request, every question, every answer checked. See `docs.typesafe.ai/cookbooks/parallel_questions`.

## Install / quickstart

```bash
git clone <repo> && cd jev-opencode-toolkit
uv sync --dev
export TYPESAFE_API_KEY=...        # environment only, read at runtime
```

```python
from jev_toolkit import Question, ask

state = "You are Jev. Answer every question with your true probability."
questions = [
    Question("is_invoice", "choice", "Is this an invoice?",
             {"yes": "It is an invoice", "no": "It is not an invoice"}),
    Question("has_signature", "noul", "Is there a handwritten signature?"),
]
r = ask(state, questions)      # ONE request
print(r.answers, r.requests)   # r.requests == 1
```

Transport: `POST https://api.typesafe.ai/v1/systemone`, header `Authorization: Bearer $TYPESAFE_API_KEY`, body `{model, state, questions}`. The key is read from the process environment **only**, at runtime; the toolkit never reads or writes any secret file. There is no live key in CI — every CI job runs the deterministic offline stub.

## opencode wiring

Edit global `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "jev": {
      "type": "local",
      "command": ["uv", "run", "--directory", "/home/meetr/jev-opencode-toolkit", "python", "-m", "jev_toolkit.mcp_server"],
      "enabled": true
    }
  }
}
```

opencode is **not** hot-reloaded — restart it after editing. The server inherits the shell environment, so `export TYPESAFE_API_KEY=...` before launching opencode; do **not** put the key in the config. Some MCP clients filter the environment before spawning servers; if your key does not reach the server, add it explicitly via the per-server `"environment"` key (the toolkit still never hardcodes it). Because opencode prefixes MCP tool names with the server name, the three tools appear as **`jev_ask`**, **`jev_ask_file`**, **`jev_calibrate`**. Tool names/descriptions are defined server-side in `mcp_server.py` (`_tool_definitions()`), so the toolkit ships **no plugin** and no `tool.definition` override.

## Capabilities

- **`ask`** — many independent typed questions (`choice` / `score` / `noul`) over one `state` in one request. Strict validation: returned answer keys must equal the requested set; probabilities must cover exactly the offered labels, be finite in `[0,1]`, and sum to 1 within `0.02`; `choice` must be the argmax of its own distribution; `choice`/`score` require `confidence`, `noul` must not carry one. Retries/backoff on `429/503/529`, honoring `Retry-After`.
- **`ask_many`** — bounded concurrency (asyncio + httpx), order-preserving, default cap **4**.
- **Cache** — keyed on `hash(state + questions + model)`; in-memory by default, optional on-disk (`JEV_CACHE_DIR`).
- **Pinned rubrics** — JSON question files with `{$bind}` slots, facts only, criteria fixed server-side.
- **Calibration lock** — `jev-lock fit` writes `decisions.lock.json`; `jev-lock check` fails CI on drift or missing calibration.
- **MCP server** — thin stdio JSON-RPC exposing exactly 3 tools: `ask`, `ask_file`, `calibrate` (surfaced by opencode as `jev_ask`, `jev_ask_file`, `jev_calibrate` because opencode prefixes MCP tool names with the server name).
- **No plugin.**

## CLI

```bash
uv run python benchmarks/bench.py --offline --out results   # writes results/RESULTS.md
uv run jev-lock check --lock calibration/decisions.lock.json \
  --questions examples/pinned_questions.json --samples calibration/samples.json
uv run ruff check . && uv run pytest -q
```

`jev-lock fit --samples calibration/samples.json --out calibration/decisions.lock.json` fits thresholds; `jev-mcp` runs the stdio server directly.

Pinned rubric (facts are bound into `{$slots}`):

```json
{
  "version": 1,
  "state": "You are Jev. Answer every independent question about the named document with your true probability. Treat absent evidence as evidence of absence.",
  "questions": {
    "is_invoice":     { "type": "choice", "instructions": "Is {$doc} an invoice?",
                        "criteria": { "yes": "It is an invoice", "no": "It is not an invoice" } },
    "has_signature":  { "type": "noul",   "instructions": "Does {$doc} contain a handwritten signature?" },
    "amount_confidence": { "type": "score",
                        "instructions": "How confident are you in the total amount stated in {$doc}?",
                        "criteria": ["very low", "low", "medium", "high", "very high"] }
  }
}
```

## Offline / CI

No key is ever needed for CI. `stub.py` mirrors the live call shape and is deterministic, with three modes: `oracle` (exact truths → perfect Brier/ECE), `miscalibrated` (overconfident; must trip the ECE detector) and `corrupt` (one malformed answer; must trip strict validation). CI runs lint + tests + build on Python 3.11/3.12, then the offline benchmark and the calibration-lock check on every push.

## How verified

- `ruff check .` and `pytest -q` (live tests carry a `live` marker and skip without `TYPESAFE_API_KEY`).
- Stub "teeth" tests: `corrupt` must raise, `miscalibrated` must fail calibration.
- Every CI run executes `benchmarks/bench.py --offline` and `jev-lock check` against the committed lock.

## Leanness

| | |
|---|---|
| Python files | **8** in `jev_toolkit/` (plus tests/benchmarks) |
| Runtime dependencies | **1** — `httpx>=0.28,<1`; stdlib only otherwise |
| MCP tools | 3 |
| Plugins shipped | 0 |
| Secret files read/written | none (env only) |
| Live calls in CI | 0 (offline stub) |

## Credits

Reuse, not rebuild. Calibration metrics (Brier, log loss, ECE, reliability) are vendored verbatim from `jev-arena` (`jev_arena/metrics.py`). The one-request fan-out and strict per-question validation patterns follow `jev-sonar/jev_sonar/jev.py`, `jev-arena/jev_arena/jev.py`, and `jev-ultrafast/jev_ultrafast/model.py` (which independently implements the same keyset/sum/argmax strictness); the offline stub's call shape follows `jev-sonar/jev_sonar/stub.py` / `jev-arena/jev_arena/stub.py`. Thanks to the TypeSafe parallel-questions cookbook and the Jev jaggedness page.

## Caveats

Text-only; no counting, math, or dates. Literal reading only, steerable by adversarial `state`, and it cannot generate text. No structural invariants are assumed (`noul` is not "choice = yes"). Confidence thresholds are *fitted* per question, not universal.
