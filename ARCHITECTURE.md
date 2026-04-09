# Entropic Atlas — Architecture Teaching Session

## Part 1: Theory — What Problem Are We Solving?

### The Competition
Berkeley RDI's **AgentX-AgentBeats** competition tests research agents on TWO benchmarks simultaneously:

1. **FieldWorkArena** — A green agent sends you an image/video/PDF from a factory, warehouse, or retail store plus a question like "How many workers are not wearing hard hats?" Your purple agent must return a precise answer.

2. **MLE-Bench** — A green agent sends you a Kaggle competition (description + data as a tar.gz file). Your purple agent must train a model and return a `submission.csv`.

Both communicate via the **A2A (Agent-to-Agent) protocol** — a standardized JSON-RPC interface where agents discover each other via agent cards.

### Why This Is Hard
- **FieldWorkArena** requires multimodal understanding (images + PDFs + videos), spatial reasoning (distances, containment, violations), and precise formatting (the answer must match exact_match, json_match, or numerical_match).
- **MLE-Bench** requires reading a competition description, choosing an ML strategy, writing complete runnable code, executing it, and handling failures — all within a 3600-second timeout.
- **Both** must be handled by a single server endpoint.

### Our Three Key Insights

1. **Structured Spatial Scene Graphs** — VLMs (GPT-4, Claude) hallucinate spatial relationships and can't count precisely. Solution: extract entities from vision → build a graph → compute distances/violations *deterministically* → feed computed facts to the LLM.

2. **Entropy-Guided Reasoning** — Not all reasoning steps are equal. Before each step, estimate which action maximizes information gain. This avoids wasting tokens on low-value reasoning and triggers reflection only when confidence is low.

3. **Self-Healing ML Pipelines** — Generated code will fail. Plan for it: execute in subprocess, capture stderr, feed error back to LLM, fix, retry up to 3 times. Always produce a submission (even a dummy one) so you never score 0.

---

## Part 2: File Interaction Map

```
                        ┌─────────────────┐
                        │    server.py     │  ← A2A entry point (Starlette + uvicorn)
                        │  Port 9019       │     Defines AgentCard with 2 skills
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │   executor.py    │  ← AgentExecutor (per-context routing)
                        │  Per-context     │     Creates Agent instances per conversation
                        │  agent pool      │     Manages task lifecycle (start → work → complete)
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │    agent.py      │  ← THE BRAIN (domain classifier + router)
                        │  _classify_domain│     Detects FieldWorkArena vs MLE-Bench
                        │  _parse_message  │     Parses A2A message into text + file parts
                        └───┬─────────┬───┘
                            │         │
              ┌─────────────▼──┐   ┌──▼──────────────┐
              │  FIELDWORK     │   │  MLEBENCH        │
              │  handler.py    │   │  handler.py      │
              │  (5-step pipe) │   │  (5-step pipe)   │
              └──┬──┬──┬──┬──┘   └──┬──┬──┬──┬─────┘
                 │  │  │  │         │  │  │  │
    ┌────────────▼┐ │  │  │    ┌────▼┐ │  │  │
    │ parser.py   │ │  │  │    │analyzer│ │  │
    │ GoalParser  │ │  │  │    │  .py │ │  │  │
    │ Extracts:   │ │  │  │    │Detects│ │  │  │
    │ - query     │ │  │  │    │task   │ │  │  │
    │ - format    │ │  │  │    │type   │ │  │  │
    └─────────────┘ │  │  │    └──────┘ │  │  │
       ┌────────────▼┐ │  │       ┌─────▼┐ │  │
       │ vision.py   │ │  │       │codegen│ │  │
       │ VisionPipe  │ │  │       │  .py  │ │  │
       │ - images    │ │  │       │Generates│ │
       │ - PDFs      │ │  │       │Python  │ │  │
       │ - videos    │ │  │       │scripts │ │  │
       │ - text      │ │  │       └───────┘ │  │
       │     │       │ │  │                 │  │
       │ detector.py │ │  │    ┌────────────▼┐ │
       │ Florence-2  │ │  │    │ executor.py  │ │
       │ (optional)  │ │  │    │ (mlebench)   │ │
       └─────────────┘ │  │    │ Subprocess   │ │
          ┌────────────▼┐ │    │ + timeout    │ │
          │ spatial.py  │ │    └──────────────┘ │
          │ Scene Graph │ │                     │
          │ - entities  │ │    ┌────────────────▼┐
          │ - relations │ │    │ strategies/     │
          │ - distances │ │    │ - tabular.py    │
          │ - violations│ │    │ - nlp.py        │
          └─────────────┘ │    │ - vision_ml.py  │
             ┌────────────▼┐   │ - timeseries.py │
             │ reasoner.py │   │ - general.py    │
             │ Uses:       │   │ - autogluon.py  │
             │ - entropy/  │   └─────────────────┘
             │   engine.py │
             └──────┬─────┘
                ┌───▼────────┐
                │formatter.py│
                │ Output fmt │
                │ matching   │
                └────────────┘

        ═══════ SHARED INFRASTRUCTURE ═══════

    ┌──────────┐  ┌──────────────┐  ┌────────────┐
    │ config.py│  │   llm.py     │  │  cost/     │
    │ All knobs│  │ LiteLLM wrap │  │ tracker.py │
    │ in one   │  │ 3 methods:   │  │ Token/cost │
    │ place    │  │ - generate() │  │ budgets    │
    │          │  │ - vision()   │  │            │
    │          │  │ - messages() │  │ router.py  │
    └──────────┘  └──────────────┘  │ 3-tier map │
                                    └────────────┘
```

---

## Part 3: File-by-File Code Walkthrough

### `src/server.py` — The Entry Point

**Purpose:** Creates the A2A Starlette server, defines the AgentCard, and starts uvicorn.

**Key lines:**
- **Lines 1-9:** Imports. `A2AStarletteApplication` is the framework that handles JSON-RPC routing. `AgentCard` describes what our agent can do.
- **Lines 34-39:** Command-line args. `--host` defaults to 127.0.0.1 (local only), `--port` to 9019.
- **Lines 41-63:** Two skills defined — one for each benchmark. These are metadata only; they tell green agents what we can do.
- **Lines 65-79:** The `AgentCard` — this is what gets served at `/.well-known/agent-card.json`. It's how other agents discover us.
- **Lines 81-84:** Wire up the `DefaultRequestHandler` with our `Executor` and an `InMemoryTaskStore`. Every incoming A2A request goes through this.
- **Lines 86-89:** Build the ASGI app and mount it.
- **Lines 102-107:** Start uvicorn. `timeout_keep_alive=300` allows long-running tasks (MLE-Bench can take minutes).

**Data flow:** HTTP request → Starlette → DefaultRequestHandler → Executor → Agent

---

### `src/executor.py` — Task Lifecycle Manager

**Purpose:** Standard AgentBeats pattern. Creates one Agent per conversation (context_id), manages start/work/complete lifecycle.

**Key concepts:**
- **Line 27:** `TERMINAL_STATES` — once a task is completed/canceled/failed, it can't be processed again.
- **Line 35:** `self.agents: dict[str, Agent]` — maps context_id to Agent instance. This allows multi-turn conversations (important for MLE-Bench's "validate" protocol).
- **Lines 38-53:** `execute()` — validates the request, creates a new task if needed, gets a TaskUpdater, and calls `agent.run()`.
- **Line 66:** `if not updater._terminal_state_reached` — the agent's `run()` method may not have set a terminal state (e.g., if it just adds artifacts). We auto-complete in that case.

---

### `src/agent.py` — THE BRAIN

**Purpose:** Receives any A2A message, classifies which benchmark sent it, and routes to the right handler.

**`_parse_message()` (lines 123-152):**
Each A2A message has `parts` — could be `TextPart`, `DataPart`, or `FilePart`. We separate them:
- Text → list of strings (the goal/instructions)
- Files → list of (name, mime_type, data) tuples (images, PDFs, competition.tar.gz)

**`_classify_domain()` (lines 154-186):**
The critical routing decision. Strategy:
1. If any file is named `competition*.tar.gz` or has gzip MIME → **MLE-Bench**
2. If text contains `# Question` AND `# Output Format` → **FieldWorkArena** (this is the exact format the FWA green agent uses)
3. If text mentions "kaggle" or "submission.csv" → **MLE-Bench**
4. Default → **FieldWorkArena** (more common in research track)

**`_handle_fieldwork()` (lines 72-89):**
Calls `FieldWorkHandler.handle()`, gets a text answer, wraps it as a `TextPart` artifact.

**`_handle_mlebench()` (lines 91-121):**
Calls `MLEBenchHandler.handle()`, gets (csv_bytes, summary). Creates TWO artifact parts:
- `TextPart` with the summary
- `FilePart` with submission.csv as base64-encoded FileWithBytes

This is critical — the green agent looks for a FilePart artifact named "submission.csv".

---

### `src/config.py` — Centralized Configuration

**Purpose:** Single source of truth for all tunable parameters.

All values can be overridden via environment variables (useful for Docker/HF Spaces).

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `fast_model` | gpt-4.1-mini | Quick tasks (parsing, classification) |
| `standard_model` | gpt-4.1 | Code generation, analysis |
| `strong_model` | gpt-4.1 | Spatial reasoning, reflection |
| `vision_model` | gpt-4.1 | Image analysis |
| `max_tokens_per_task` | 150,000 | Budget per task |
| `max_reflection_rounds` | 2 | How many times to self-reflect |
| `max_video_frames` | 30 | Max frames to extract from video |
| `code_execution_timeout` | 600s | Max time for ML pipeline |
| `max_code_iterations` | 3 | Max fix-and-retry attempts |

---

### `src/llm.py` — Unified LLM Interface

**Purpose:** Wraps litellm for multi-provider model access.

Three methods, all tracking cost:
- `generate(prompt, model_tier, ...)` — Text in, text out. Used everywhere.
- `vision_analyze(image_bytes, prompt, ...)` — Image + text in, text out. Used by vision.py.
- `generate_with_messages(messages, ...)` — Full message array control for multi-turn.

**Why litellm?** It provides a unified interface to OpenAI, Anthropic, Ollama, HuggingFace, and 100+ other providers. Change model by changing the string prefix: `openai/gpt-4.1`, `anthropic/claude-sonnet-4-20250514`, `ollama/qwen2.5-vl:7b`.

---

### `src/fieldwork/parser.py` — Goal String Parser

**Purpose:** Parse the structured goal format from FieldWorkArena's green agent.

The green agent sends:
```
# Question
How many workers are wearing hard hats?

# Input Data
image1.jpg image2.jpg

# Output Format
number
```

`GoalParser.parse()` uses a regex (`^# (Question|Input Data|Output Format)$`) to split into sections, producing a `FieldWorkTask(query, input_files, output_format)`.

Fallback: if no structured sections found, the entire text becomes the query.

---

### `src/fieldwork/vision.py` — Multimodal File Processing

**Purpose:** Convert any file attachment (image, PDF, video, text) into rich text context.

**Image processing (lines 79-130):**
1. Decode base64 → raw bytes
2. Convert to RGB JPEG (normalize format)
3. **NEW: Run Florence-2 local detector** (if available) — extracts object counts, bounding boxes, PPE status
4. Send to GPT-4.1 vision with `VISION_PROMPT` — a detailed prompt asking for positions, PPE, distances, violations
5. If Florence-2 detected objects, inject those counts into the VLM prompt: "A detection model has already identified the following objects. Use these EXACT counts."

**PDF processing (lines 100-112):**
Uses pypdf to extract text page by page. Simple but effective for text-heavy PDFs.

**Video processing (lines 114-139):**
1. Write bytes to temp file
2. OpenCV extracts frames at 2-second intervals (configurable)
3. Select key frames (max 10, evenly spaced)
4. Send each key frame to vision model
5. Concatenate all frame descriptions

---

### `src/fieldwork/detector.py` — Local Object Detection (Florence-2)

**Purpose:** Address VLMs' known weakness at counting objects.

**The problem:** If you ask GPT-4 "How many workers are in this image?", it might say 3 when there are 5. VLMs are notoriously bad at precise counting.

**The solution:** Run a dedicated detection model FIRST to get exact counts, bounding boxes, and PPE status. Then tell the VLM: "There are exactly 5 workers. 3 have hard hats, 2 don't."

**Architecture:**
- Lazy-loads Microsoft Florence-2-base on first use
- Falls back gracefully if torch/transformers unavailable (API-only mode)
- Runs two inference tasks: `<MORE_DETAILED_CAPTION>` and `<OD>` (object detection)
- Extracts PPE keywords from detected labels (hard hat, vest, goggles, gloves, mask)
- Returns `DetectionResult` with structured `object_counts`, `ppe_detected`, `raw_caption`

---

### `src/fieldwork/spatial.py` — The Crown Jewel

**Purpose:** Build a queryable spatial scene graph from vision descriptions.

**Why this matters:** FieldWorkArena asks questions like "Which workers are within 3 meters of the forklift?" or "Are there any PPE violations in the loading dock?" LLMs can't reliably answer these from raw text. But if we extract entities with positions and compute distances deterministically, we can.

**Data structures:**
```python
SpatialEntity(id, label, position=(x,y), attributes={}, zone="")
SpatialRelation(subject, predicate, object, distance=None)
SpatialScene(entities, relations, safety_rules)
```

**Key methods:**
- `compute_distance(e1, e2)` → Euclidean distance, rounded to spatial_precision
- `compute_all_distances()` → fills in distance for every relation
- `query_near(entity_id, radius)` → returns all entities within radius
- `check_constraints()` → checks safety rules (PPE, distance-based)
- `to_fact_sheet()` → converts everything to text: "Entity w1 (worker) at (2.0, 3.0) in zone loading_dock. Distance w1→f1: 5.0 units."

**SpatialAnalyzer.build_scene():**
1. Send file contexts + query to LLM
2. Ask LLM to return JSON: `{entities: [...], relations: [...], safety_rules: [...]}`
3. Parse into SpatialEntity/SpatialRelation objects
4. Call `compute_all_distances()` to fill in deterministic measurements
5. Return the complete scene

The key insight is: the LLM extracts *what* and *where*, but the *math* is done by Python. This eliminates hallucinated distances.

---

### `src/fieldwork/reasoner.py` — Entropy-Guided Reasoning

**Purpose:** Combine all evidence + spatial facts to produce the final answer.

**Flow:**
1. Join all file_contexts into one evidence string
2. Get spatial facts from scene.to_fact_sheet()
3. Build a comprehensive prompt with question + evidence + spatial facts + format
4. Send to strong model with REASONING_SYSTEM_PROMPT
5. **Entropy check:** estimate confidence (0.0-1.0)
6. If confidence < 0.6 → call `_refine_answer()` for self-reflection

The system prompt emphasizes: "Use computed spatial analysis when available (these are deterministic calculations)." This steers the LLM to trust our computed distances over its own guesses.

---

### `src/fieldwork/formatter.py` — Output Format Matching

**Purpose:** The green agent's evaluation is strict — exact_match means EXACT. This module ensures our answer matches.

**Format handlers:**
- `json` → Extract JSON from answer text, validate it parses
- `number`/`integer`/`numeric` → Extract the first number via regex
- `yes/no`/`boolean` → Normalize to exactly "yes" or "no"
- `list` → Clean bullet points to comma-separated
- Everything else → Strip markdown formatting (bold, code fences)

This is the difference between scoring 0.0 and 1.0 on many tasks.

---

### `src/entropy/engine.py` — Information Gain Estimation

**Purpose:** The "entropic" in Entropic Atlas.

**`select_best_action()`:**
Given multiple candidate actions, ask the fast model to rate each on information gain (1-10). Pick the highest. This is used sparingly — only when there are genuinely multiple paths to explore.

**`estimate_confidence()`:**
Given an answer + evidence + query, ask the fast model: "How confident are you?" Returns 0.0-1.0. Used by the reasoner to decide whether to reflect.

**Why "fast" model?** These are meta-reasoning calls — they guide the reasoning but don't produce the final answer. Using the fast model (gpt-4.1-mini) keeps costs low while the strong model does the heavy lifting.

---

### `src/mlebench/handler.py` — ML Pipeline Orchestrator

**Purpose:** End-to-end Kaggle competition solving.

**5-step pipeline:**
1. **Extract:** Find competition.tar.gz in file attachments, extract to temp directory
2. **Analyze:** Read description.md, list data files, preview CSVs, determine task type
3. **Generate:** Create a complete Python script using strategy template + LLM
4. **Execute:** Run in subprocess with timeout, capture stdout/stderr
5. **Self-heal:** If execution fails, feed error to LLM, fix code, retry (up to 3 times)

**Fallback:** If all iterations fail, generate a dummy submission (correct format, dummy values). This ensures we never score 0 — even a bad submission might get partial credit.

---

### `src/mlebench/codegen.py` — ML Code Generator

**Purpose:** Generate complete, runnable Python scripts for any Kaggle competition.

The prompt gives the LLM: competition description, file listing, data preview, analysis results, and a strategy template. The LLM fills in the specifics (column names, preprocessing, model hyperparameters).

**Self-healing:** `fix()` takes the failed code + error + stdout and asks the LLM to fix it. Common fixes: wrong column name, missing file, library not available, wrong dtype.

---

### `src/cost/tracker.py` and `router.py`

**tracker.py:** Accumulates prompt_tokens, completion_tokens, num_calls, estimated_cost_usd across all LLM calls. `has_budget()` checks against max_tokens.

**router.py:** Maps task types to model tiers:
- Fast tasks (classify, parse, format) → gpt-4.1-mini
- Standard tasks (code_gen, analyze, reason) → gpt-4.1
- Strong tasks (spatial_reasoning, reflection, complex_vision) → gpt-4.1

---

## Part 4: Request Lifecycle — End to End

### FieldWorkArena Example: "How many workers are wearing hard hats?"

```
Green Agent                    Purple Agent (Entropic Atlas)
     │                                    │
     │ POST / (A2A JSON-RPC)              │
     │ TextPart: "# Question..."          │
     │ FilePart: warehouse.jpg (base64)   │
     │──────────────────────────────────►  │
     │                                    │
     │                         server.py receives
     │                         executor.py creates Agent
     │                         agent.py._parse_message()
     │                           → text_parts, file_parts
     │                         agent.py._classify_domain()
     │                           → "fieldwork" (has # Question)
     │                                    │
     │                         fieldwork/handler.py.handle()
     │                           1. parser.parse(text)
     │                              → FieldWorkTask(query="How many...")
     │                           2. vision.process_file("warehouse.jpg")
     │                              → detector.detect() [Florence-2]
     │                                 → 7 workers, 5 hard hats
     │                              → llm.vision_analyze()
     │                                 → "7 workers visible, 5 with hats..."
     │                           3. spatial.build_scene()
     │                              → SpatialScene with 7 entities
     │                              → compute_all_distances()
     │                              → check_constraints()
     │                                 → ["2 workers missing PPE"]
     │                           4. reasoner.reason()
     │                              → "5 workers are wearing hard hats"
     │                              → entropy.estimate_confidence()
     │                                 → 0.85 (high, no reflection needed)
     │                           5. formatter.format_answer("5", "number")
     │                              → "5"
     │                                    │
     │  ◄──────────────────────────────── │
     │  TaskArtifactUpdateEvent           │
     │  TextPart: "5"                     │
     │                                    │
     │  Green agent evaluates:            │
     │  numerical_match("5", "5") → 1.0   │
```

### MLE-Bench Example: Spaceship Titanic

```
Green Agent                    Purple Agent
     │                                    │
     │ TextPart: instructions.txt         │
     │ FilePart: competition.tar.gz       │
     │──────────────────────────────────►  │
     │                                    │
     │                         _classify_domain()
     │                           → "mlebench" (has tar.gz)
     │                                    │
     │                         mlebench/handler.py.handle()
     │                           1. Extract tar → /tmp/atlas_mle_xyz/
     │                              → home/data/train.csv, test.csv,
     │                                description.md
     │                           2. analyzer.analyze()
     │                              → tabular_classification, accuracy
     │                           3. codegen.generate()
     │                              → AutoGluon script (200 lines)
     │                           4. executor.execute()
     │                              → subprocess runs pipeline.py
     │                              → [SUCCESS] submission.csv produced
     │                           5. Return (csv_bytes, summary)
     │                                    │
     │  ◄──────────────────────────────── │
     │  TextPart: "Strategy: tabular..."  │
     │  FilePart: submission.csv (base64) │
     │                                    │
     │  grade_csv(submission.csv) → 0.79  │
```

---

## Part 5: Key Design Decisions

### Why A2A instead of a simpler API?
A2A is the competition standard. All green agents communicate via A2A. By following the protocol exactly, our agent works with any evaluator without modification.

### Why LiteLLM instead of direct OpenAI SDK?
LiteLLM abstracts the provider. We can switch from OpenAI to Anthropic to a local Ollama model by changing one string in config. This is critical for cost optimization and experimentation.

### Why deterministic spatial computation instead of asking the LLM?
LLMs hallucinate distances. If you ask "How far is the worker from the forklift?", the LLM might say "about 2 meters" when it's actually 5. By extracting positions (which LLMs are decent at) and computing distances (which Python does perfectly), we get exact answers.

### Why AutoGluon over simpler models?
AutoGluon consistently wins Kaggle tabular competitions by ensembling multiple model types. Our strategy template uses AutoGluon with a 5-minute budget, falling back to LightGBM if AutoGluon isn't available.

### Why self-healing code over perfect generation?
It's impossible to generate bug-free code for arbitrary Kaggle competitions on the first try. The dataset might have unexpected column names, missing values, or format quirks. Self-healing (read error → fix → retry) is more robust than trying to be perfect.
