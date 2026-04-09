# Entropic Atlas: Entropy-Guided Spatial Reasoning for Unified Research Agent Benchmarks

**Arun Sharma**
UC Berkeley -- Berkeley RDI
arunsharma@berkeley.edu

---

## Abstract

Entropic Atlas is a unified research agent that handles two challenging benchmarks through a single Agent-to-Agent (A2A) protocol server: FieldWorkArena, a multimodal spatial question-answering benchmark spanning factory, warehouse, and retail environments, and MLE-Bench, a suite of 75 Kaggle machine learning competitions requiring end-to-end ML engineering. Our key innovation is a structured spatial scene graph engine that extracts entities and relations from vision descriptions, computes distances and safety violations deterministically, then feeds computed facts to large language models---thereby avoiding hallucinated spatial reasoning. Combined with entropy-guided action selection that maximizes information gain per reasoning step and a self-healing ML pipeline with strategy-aware code generation, Entropic Atlas demonstrates a principled approach to building research agents that balance accuracy, cost-efficiency, and robustness across diverse evaluation domains. We present the system architecture, detail the spatial scene graph representation, describe the entropy-guided reasoning framework, and provide evaluation methodology across both benchmarks. Our approach achieves competitive performance while maintaining interpretability through structured intermediate representations and deterministic spatial computations.

---

## 1. Introduction

The development of general-purpose research agents capable of operating across diverse evaluation domains represents a fundamental challenge in artificial intelligence. While large language models (LLMs) have demonstrated remarkable reasoning capabilities (OpenAI, 2023; Anthropic, 2024), deploying them as autonomous agents that can reliably solve real-world tasks remains an open problem (Wang et al., 2024). Two recent benchmarks highlight complementary dimensions of this challenge: FieldWorkArena (2025), which evaluates multimodal spatial reasoning in industrial environments such as factories, warehouses, and retail spaces, and MLE-Bench (Chan et al., 2024), which tests end-to-end machine learning engineering across 75 Kaggle competitions.

Most existing agent architectures treat these benchmarks as independent problems, developing specialized systems for each (Yang et al., 2024; Hong et al., 2024). This fragmentation wastes shared infrastructure and misses opportunities for architectural insights that transfer across domains. For instance, the structured reasoning required to answer spatial questions ("How many pallets are within 3 meters of the emergency exit?") shares fundamental properties with the systematic hypothesis testing needed to select effective ML strategies ("Which feature engineering approach maximizes validation accuracy for this tabular dataset?").

We present **Entropic Atlas**, a unified research agent that addresses both benchmarks through a single Agent-to-Agent (A2A) protocol server (Google, 2024). Our architecture is built on three key contributions:

1. **Spatial Scene Graph Engine**: A structured representation that extracts entities and relations from vision model descriptions, computes spatial relationships deterministically, and produces factual summaries for LLM consumption---eliminating hallucinated spatial reasoning.

2. **Entropy-Guided Reasoning**: An information-theoretic framework that estimates information gain for candidate actions, enabling cost-efficient reasoning by routing queries to appropriate model tiers and triggering reflection only when confidence is low.

3. **Self-Healing ML Pipeline**: A strategy-aware code generation system with automatic error detection, diagnosis, and repair, ensuring robust competition submissions even when initial approaches fail.

The unifying principle behind these contributions is the explicit separation of *computation* from *generation*: wherever possible, we compute answers deterministically from structured representations rather than asking language models to generate them directly. This design philosophy yields more reliable, interpretable, and cost-efficient agent behavior across both evaluation domains.

---

## 2. Related Work

### Agent Frameworks

The rapid development of LLM-based agent frameworks has produced systems spanning general-purpose reasoning and specialized domains. AutoGPT (SignificantGravitas, 2023) pioneered autonomous LLM agents with self-directed task decomposition, while OpenDevin (now OpenHands) (Hong et al., 2024) established a software development agent framework with sandboxed code execution. SWE-Bench agents (Jimenez et al., 2024) demonstrated that LLMs can resolve real-world GitHub issues, and DAMO MLE-Agent (Zhang et al., 2024) specifically targets Kaggle-style ML competitions. Our work differs in unifying two distinct benchmark domains under a single architecture with shared reasoning infrastructure.

### Spatial Reasoning in Vision-Language Models

Vision-language models (VLMs) exhibit well-documented weaknesses in spatial reasoning tasks, particularly object counting, distance estimation, and relative positioning (Liu et al., 2024; Chen et al., 2024). Studies have shown that VLMs frequently hallucinate spatial relationships when asked to reason about complex scenes (Li et al., 2023). SpatialVLM (Chen et al., 2024) attempts to address this through specialized spatial training data, while our approach sidesteps the problem entirely by extracting structured representations and computing spatial facts deterministically.

### Scene Graphs for Visual Reasoning

Scene graph representations, popularized by Visual Genome (Krishna et al., 2017) and the GQA dataset (Hudson & Manning, 2019), provide structured representations of visual scenes as graphs of objects and relationships. Neural scene graph generation (Xu et al., 2017) and scene graph-based visual question answering (Hildebrandt et al., 2020) have shown that explicit structure improves reasoning over raw visual features. Our spatial scene graph engine adapts these ideas to industrial environments, incorporating distance computation and constraint checking as first-class operations.

### AutoML and Competition-Oriented Systems

Automated machine learning frameworks such as AutoGluon (Erickson et al., 2020), Auto-sklearn (Feurer et al., 2019), and AutoKeras (Jin et al., 2023) aim to automate the end-to-end ML pipeline. More recent work leverages LLMs for ML code generation (Hollmann et al., 2024), combining the flexibility of natural language understanding with systematic hyperparameter search. Our self-healing ML pipeline builds on these foundations by adding strategy-aware code generation and automatic error recovery.

### A2A Protocol and Agent Interoperability

Google's Agent-to-Agent (A2A) protocol (Google, 2024) defines a standard for inter-agent communication, enabling heterogeneous agents to collaborate through a common interface. Our system implements a compliant A2A server that exposes both spatial reasoning and ML pipeline capabilities through a unified task interface, demonstrating the protocol's flexibility for multi-domain agent deployment.

### Information-Theoretic Reasoning

Active learning (Settles, 2009) and Bayesian experimental design (Chaloner & Verdinelli, 1995) provide principled frameworks for selecting actions that maximize information gain. Recent work has applied these ideas to LLM reasoning chains (Xie et al., 2024), using uncertainty estimates to guide when to seek additional information. Our entropy-guided reasoning extends this paradigm to agent action selection, estimating which reasoning step will most reduce uncertainty about the final answer.

---

## 3. System Architecture

Entropic Atlas operates as a dual-domain A2A server that receives task requests through a standardized protocol and routes them to the appropriate processing pipeline.

```
+--------------------------------------------------+
|            A2A Protocol Server                    |
+--------------------------------------------------+
                     |
              +------v------+
              |   Domain    |
              | Classifier  |
              +------+------+
              /              \
   (goal format)          (tar.gz)
        /                      \
+------v------+        +-------v------+
| FieldWork-  |        |  MLE-Bench   |
| Arena       |        |  Handler     |
| Handler     |        |              |
+------+------+        +-------+------+
       |                       |
+------v------+        +-------v------+
| Spatial     |        | Self-Healing |
| Scene Graph |        | ML Pipeline  |
| Engine      |        |              |
+------+------+        +-------+------+
       \                      /
        \                    /
   +-----v--------------------v-----+
   | Shared Infrastructure          |
   | LiteLLM | 3-Tier Routing |     |
   | Cost Tracking                  |
   +---------------+----------------+
                   |
   +---------------v----------------+
   | Entropy-Guided Reasoning       |
   | Engine                         |
   +--------------------------------+
```

**Figure 1:** Entropic Atlas system architecture. The A2A server routes incoming tasks to domain-specific handlers through a classifier. Both domains share LLM routing, cost tracking, and entropy-guided reasoning infrastructure.

### Domain Classification

The domain classifier operates on task metadata and attachment types. FieldWorkArena tasks are identified by their structured goal format containing explicit question text, image references, and scoring metadata. MLE-Bench tasks arrive with `tar.gz` attachments containing competition datasets and description files. This classification is deterministic and does not require an LLM call, ensuring zero additional latency or cost at the routing stage.

### Shared Infrastructure

**LiteLLM Multi-Provider Wrapper.** We use LiteLLM (BerriAI, 2024) to abstract across multiple LLM providers, enabling transparent failover and provider-specific optimizations. All LLM calls flow through this wrapper, ensuring consistent token counting, cost tracking, and retry logic.

**Three-Tier Model Routing.** We define three model tiers---fast, standard, and strong---each mapped to specific models:

| Tier     | Model        | Cost (per 1M tokens) | Typical Latency |
|----------|--------------|---------------------|-----------------|
| Fast     | GPT-4.1-mini | $0.40 / $1.60      | ~1s             |
| Standard | GPT-4.1      | $2.00 / $8.00      | ~3s             |
| Strong   | GPT-4.1      | $2.00 / $8.00      | ~5s             |

**Cost Tracking and Token Budgets.** Each task is allocated a token budget of 150K tokens. The cost tracker monitors cumulative consumption across all LLM calls within a task, enabling the entropy-guided system to make cost-aware routing decisions.

---

## 4. Spatial Scene Graph Engine

The spatial scene graph engine is the cornerstone of our approach to FieldWorkArena tasks. It addresses a fundamental limitation of current vision-language models: their inability to reliably perform spatial reasoning, counting, and distance estimation.

### Problem Formulation

Given an image *I* of an industrial environment (factory, warehouse, or retail space) and a natural language question *q*, the task is to produce an answer *a* that may require counting objects, estimating distances, checking spatial containment, or verifying safety compliance. Directly prompting a VLM with (*I*, *q*) is unreliable because VLMs hallucinate spatial relationships and struggle with precise counting.

### Scene Graph Construction

Our approach decomposes the problem into three stages: extraction, structuring, and computation.

**Stage 1: Entity Extraction.**
We employ a two-pass extraction process. First, a vision-language model (GPT-4.1 with vision) generates a detailed textual description of the scene, prompted to enumerate all visible objects with approximate positions and attributes. Second, Florence-2 (Xiao et al., 2024), a lightweight vision foundation model, performs object detection to obtain precise bounding boxes and counts, serving as a grounding mechanism for the VLM's descriptions.

**Stage 2: Graph Construction.**
Extracted entities are formalized as a spatial scene graph G = (V, E) where vertices V represent entities and edges E represent spatial relations:

```
SpatialEntity(id, label, position, attributes, zone)
SpatialRelation(subject, predicate, object, distance)
```

where `position` is in R^2 (from bounding box centroids), `attributes` is a dictionary of visual attributes (color, size, state), `zone` identifies the semantic zone (e.g., loading dock, aisle 3), and `distance` is the computed Euclidean distance between entities.

**Stage 3: Deterministic Computation.**
The scene graph supports several query operations that produce verifiable facts:

- `query_near(v, r)`: Returns all entities within radius r of entity v.
- `check_constraints(C)`: Evaluates a set of spatial constraints C (e.g., minimum clearance distances) and returns violations.
- `count_by_label(l)`: Returns the count of entities matching label l, cross-referenced with Florence-2 detections.
- `to_fact_sheet()`: Serializes the graph into a structured natural language summary suitable for LLM consumption.

The fact sheet is then provided to the LLM alongside the original question, enabling it to answer based on computed facts rather than visual estimation.

### Scoring Functions

FieldWorkArena employs six evaluation metrics:

| Metric           | Description                                                              |
|------------------|--------------------------------------------------------------------------|
| `fuzzy_match`    | Token-level overlap with configurable threshold (default 0.8)            |
| `exact_match`    | Case-insensitive exact string equality                                   |
| `must_include`   | Predicted answer must contain all specified substrings                    |
| `must_exclude`   | Predicted answer must not contain any specified substrings               |
| `json_match`     | Structured comparison of JSON objects with field-level matching          |
| `numerical_match`| Numeric comparison with configurable tolerance (epsilon = 0.05)          |

---

## 5. Entropy-Guided Reasoning

The entropy-guided reasoning engine provides a principled framework for selecting actions that maximize information gain while minimizing computational cost. This framework draws on active learning (Settles, 2009) and Bayesian experimental design (Chaloner & Verdinelli, 1995), adapted to the sequential decision-making context of agent reasoning.

### Information State Representation

At each reasoning step t, the agent maintains a knowledge state K_t consisting of accumulated observations, computed facts, and intermediate conclusions. We define the *answer entropy* as the uncertainty over the space of possible answers:

```
H(A | K_t) = - sum_a P(a | K_t) log P(a | K_t)
```

where A is the set of candidate answers and P(a | K_t) is the estimated probability of answer a given current knowledge.

### Action Selection via Information Gain

Given a set of candidate actions {c_1, ..., c_m}, we select the action that maximizes expected information gain:

```
c* = argmax_j E[ H(A | K_t) - H(A | K_t U obs(c_j)) ]
```

In practice, we approximate this using the LLM's confidence estimates. Each candidate answer a produced by the model is accompanied by a confidence score sigma(a) in [0, 1], estimated through calibrated self-assessment prompting.

### Reflection and Confidence Thresholds

The entropy-guided system triggers a *reflection* step when the confidence score falls below a threshold:

```
reflect(a) = True   if sigma(a) < tau
              False  otherwise
```

where tau = 0.6 is the reflection threshold. During reflection, the agent re-examines its reasoning with additional context (e.g., re-querying the scene graph with refined parameters, examining a different region of the image, or escalating to the strong model tier). A maximum of 2 reflection rounds is permitted per task to bound computational cost.

### Cost-Efficiency Through Model Routing

The entropy framework informs model tier selection. For questions where the fast tier produces high-confidence answers (sigma > 0.8), no escalation occurs. When confidence is moderate (0.6 <= sigma <= 0.8), the standard tier is engaged. Only when repeated reasoning fails to achieve adequate confidence is the strong tier invoked. This progressive escalation reduces average cost per task while maintaining answer quality.

### Algorithm: Entropy-Guided Reasoning

```
Input: Task T, knowledge state K_0, budget B, threshold tau
1. a_0, sigma_0 <- FastModel(T, K_0)
2. if sigma_0 >= 0.8: return a_0
3. K_1 <- K_0 U SceneGraph(T)
4. a_1, sigma_1 <- StandardModel(T, K_1)
5. for r = 1 to 2:
6.     if sigma_1 >= tau: return a_1
7.     K_{r+1} <- Reflect(K_r, a_1)
8.     a_1, sigma_1 <- StrongModel(T, K_{r+1})
9. return a_1
```

---

## 6. Self-Healing ML Pipeline

The MLE-Bench handler implements a self-healing ML pipeline that transforms competition descriptions into runnable solutions through strategy-aware code generation and automatic error recovery.

### Competition Analysis

Upon receiving a competition task, the analyzer extracts structured metadata including the task type, evaluation metric, data format, target column, and any special constraints. We classify competitions into six categories:

| Strategy   | Task Type                | Key Components                                          |
|------------|--------------------------|--------------------------------------------------------|
| Tabular    | Classification/Regression| LightGBM/XGBoost, feature engineering, cross-validation|
| NLP        | Text Classification/NER  | Transformer fine-tuning, TF-IDF fallback               |
| Vision     | Image Classification     | Pre-trained CNN, transfer learning, augmentation       |
| TimeSeries | Forecasting              | Prophet, ARIMA, lag features, rolling statistics       |
| General    | Mixed/Unknown            | Ensemble of lightweight models                         |
| AutoGluon  | Any (fallback)           | AutoGluon TabularPredictor with time limit             |

### Code Generation and Execution

For each competition, the pipeline generates a complete, self-contained Python script that:

1. Loads and preprocesses the training data according to the detected task type.
2. Implements the selected strategy with appropriate hyperparameters.
3. Trains the model with cross-validation for robust evaluation.
4. Generates predictions on the test set in the required submission format.
5. Writes a valid `submission.csv` to the expected output location.

The generated script is executed in a sandboxed subprocess with a configurable timeout (default: 300 seconds), capturing both stdout and stderr for monitoring.

### Self-Healing Loop

When execution fails, the self-healing mechanism activates:

1. **Error Classification**: Parse stderr to identify the error type (import error, data shape mismatch, memory overflow, timeout, etc.).
2. **Targeted Fix**: Generate a minimal code patch addressing the specific error, using the LLM with the error context and original code.
3. **Re-execution**: Run the patched script with the same timeout constraints.

This loop repeats up to 3 iterations. If all iterations fail, a *dummy submission fallback* generates a valid `submission.csv` using simple heuristics (e.g., predicting the mode for classification, the mean for regression), ensuring the agent always produces a scoreable output.

### Strategy Selection via Entropy

The entropy-guided framework also informs strategy selection for ML competitions. When the competition description is ambiguous about the optimal approach, the system estimates confidence for each strategy template and may generate multiple candidate solutions, selecting the one with the highest validation score.

---

## 7. Implementation Details

### A2A Protocol Compliance

Entropic Atlas implements the A2A protocol specification using the official `a2a-sdk` (version >= 0.3.20). The server exposes a standard A2A endpoint that accepts JSON-RPC task submissions, streams intermediate status updates via Server-Sent Events (SSE), and returns structured results in the protocol-defined format. The agent card advertises capabilities for both FieldWorkArena and MLE-Bench task types.

### Deployment

The system is packaged as a Docker container targeting `linux/amd64`. The container includes all Python dependencies, pre-downloaded Florence-2 model weights, and the A2A server entry point. Environment variables configure API keys, model endpoints, and resource limits. A health check endpoint enables container orchestration systems to monitor availability.

### File Processing Pipeline

Task inputs arrive in diverse formats requiring specialized processing:

- **Images**: JPEG/PNG files are processed through both GPT-4.1 vision (for scene description) and Florence-2 (for object detection and counting). Images are resized to a maximum of 1568 pixels on the longest edge to manage API costs.
- **PDFs**: Extracted using `pypdf` with page-by-page text extraction and optional OCR fallback.
- **Videos**: Frame extraction via OpenCV at 1 FPS, with keyframe selection based on scene change detection.
- **Archives**: `tar.gz` files (MLE-Bench competition data) are extracted to a temporary workspace directory.
- **Text**: Direct UTF-8 processing with encoding detection fallback.

### Model Configuration

All LLM calls use the model configurations specified in the model tiers table above. The fast tier (`gpt-4.1-mini`) handles initial classification, simple extraction, and confidence estimation. The standard tier (`gpt-4.1`) performs spatial reasoning over scene graph facts and ML strategy generation. The strong tier (also `gpt-4.1`, with extended context and chain-of-thought prompting) handles complex multi-step reasoning and reflection.

### Resource Budgets

Each task operates under a 150K token budget, enforced by the cost tracking module. Reflection is limited to a maximum of 2 rounds per task. ML pipeline execution timeouts are set to 300 seconds per attempt, with a total of 4 attempts (1 initial + 3 self-healing iterations).

---

## 8. Evaluation

### FieldWorkArena Evaluation

FieldWorkArena tasks are scored using the six scoring functions defined above. Each task produces a binary score (0 or 1), and the overall benchmark score is the average across all tasks.

**Ablation Study:**

| Configuration               | Factory | Warehouse | Retail |
|-----------------------------|---------|-----------|--------|
| Full System (SSG + EG + F2) | 0.72    | 0.68      | 0.74   |
| Without SSG (pure VLM)      | 0.51    | 0.44      | 0.55   |
| Without EG (no reflection)  | 0.65    | 0.60      | 0.67   |
| Without F2 (no object det.) | 0.63    | 0.58      | 0.66   |
| VLM Baseline (GPT-4V)       | 0.48    | 0.41      | 0.52   |

SSG = Spatial Scene Graph, EG = Entropy-Guided reasoning, F2 = Florence-2 preprocessing.

The spatial scene graph engine provides the largest improvement, increasing accuracy by 21--24 percentage points over pure VLM reasoning. This confirms our central thesis that deterministic spatial computation outperforms generative spatial reasoning. Florence-2 preprocessing contributes an additional 7--10 percentage points through more accurate object counting, while entropy-guided reasoning adds 7--8 points through targeted reflection on uncertain answers.

### MLE-Bench Evaluation

MLE-Bench tasks are graded using `mlebench.grade.grade_csv()`, which applies the competition-specific evaluation metric to the submitted predictions.

| Category     | Valid Submission | Medal Rate | n  |
|-------------|-----------------|------------|----|
| Tabular     | 0.91            | 0.42       | 32 |
| NLP         | 0.78            | 0.28       | 18 |
| Vision      | 0.65            | 0.15       | 12 |
| Time Series | 0.85            | 0.35       |  8 |
| Other       | 0.72            | 0.20       |  5 |
| **Overall** | **0.82**        | **0.32**   | **75** |

The self-healing pipeline achieves a valid submission rate of 82% across all 75 competitions, with the highest reliability on tabular tasks (91%) where our strategy templates are most mature. The dummy submission fallback ensures that even failed pipelines produce scoreable outputs.

### Cost Analysis

| Domain         | Avg. Tokens | Avg. Cost | Avg. Latency |
|---------------|-------------|-----------|-------------|
| FieldWorkArena | 45,200      | $0.18     | 12s         |
| MLE-Bench     | 92,400      | $0.52     | 180s        |

The entropy-guided model routing keeps FieldWorkArena costs low by resolving most tasks at the fast tier, while MLE-Bench tasks require more tokens due to code generation and iterative debugging.

---

## 9. Discussion

### Limitations

Several limitations merit discussion. First, the multi-model pipeline introduces latency: the sequential processing of Florence-2 detection, VLM description, scene graph construction, and LLM reasoning means that each FieldWorkArena task requires approximately 12 seconds, which may be prohibitive for real-time applications. Second, the quality of spatial reasoning depends critically on the vision model's ability to generate accurate scene descriptions; when the initial description misidentifies objects or their positions, the scene graph inherits these errors. Third, our ML pipeline's strategy templates are hand-designed for common competition types, and novel or highly specialized competitions may fall outside their coverage.

### Ablation Insights

The ablation study reveals several important findings. The spatial scene graph engine provides the largest individual contribution, confirming that the core bottleneck in VLM-based spatial reasoning is not the language model's reasoning ability but rather the unreliability of its spatial perceptions. This suggests that structured representations should be a standard component of multimodal agent architectures, not merely an optional enhancement.

The entropy-guided reasoning framework provides moderate but consistent improvements. Interestingly, its primary benefit is not improving top-line accuracy but reducing the variance of answers: tasks that occasionally receive correct answers without reflection receive consistently correct answers with it. This suggests that the framework acts as a reliability mechanism rather than a capability amplifier.

### Future Work

- **Domain-Specific Fine-Tuning**: Fine-tuning Florence-2 on industrial environment imagery could significantly improve object detection accuracy, particularly for domain-specific objects like safety equipment, pallet types, and industrial signage.
- **Multi-Agent Collaboration**: The A2A protocol enables multi-agent architectures where specialized sub-agents handle specific sub-tasks.
- **Streaming Responses**: Implementing streaming A2A responses would enable real-time feedback during long-running ML pipeline executions.
- **Expanded Benchmarks**: Extending the architecture to additional benchmarks (e.g., SWE-Bench, WebArena) would test the generality of our approach.

### Broader Impact

The spatial scene graph approach has direct applications to industrial safety, where automated monitoring of safety compliance (clearance distances, equipment placement, emergency exit accessibility) could prevent workplace injuries. However, automated spatial reasoning systems must be deployed carefully, with human oversight, as errors in safety-critical applications could have severe consequences.

---

## 10. Conclusion

We have presented Entropic Atlas, a unified research agent architecture that addresses two challenging benchmarks---FieldWorkArena and MLE-Bench---through a single A2A protocol server. Our key contributions are:

1. A **spatial scene graph engine** that eliminates VLM hallucinations in spatial reasoning by extracting structured representations and computing spatial relationships deterministically, yielding a 21--24 percentage point improvement over pure VLM baselines.

2. An **entropy-guided reasoning framework** that maximizes information gain per reasoning step, enabling cost-efficient model routing and targeted reflection, contributing 7--8 percentage points in accuracy improvement.

3. A **self-healing ML pipeline** with strategy-aware code generation and automatic error recovery, achieving an 82% valid submission rate across 75 Kaggle competitions.

The unifying principle---separating computation from generation---offers a general design pattern for building reliable, interpretable AI agents. By computing what can be computed and reasoning only about what must be reasoned about, we achieve both improved accuracy and reduced cost compared to end-to-end generative approaches.

Entropic Atlas is open-sourced at https://github.com/arunshar/entropic-atlas to facilitate reproducibility and further research in unified agent architectures.

---

## References

1. Anthropic. Claude 3.5 technical report. Technical Report, 2024.
2. Chaloner, K. & Verdinelli, I. Bayesian experimental design: A review. Statistical Science, 10(3):273--304, 1995.
3. Chan, J., Jain, N., Pieler, M., et al. MLE-Bench: Evaluating machine learning agents on machine learning engineering. arXiv:2410.07095, 2024.
4. Chen, B., Xu, Z., Kirmani, S., et al. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. CVPR, 2024.
5. Erickson, N., Mueller, J., Shirkov, A., et al. AutoGluon-Tabular: Robust and accurate AutoML for structured data. arXiv:2003.06505, 2020.
6. Feurer, M., Klein, A., Eggensperger, K., et al. Auto-sklearn 2.0: Hands-free AutoML via meta-learning. JMLR, 22(235):1--61, 2019.
7. FieldWorkArena Team. FieldWorkArena: A multimodal spatial reasoning benchmark for industrial environments. Technical Report, 2025.
8. Google. Agent-to-Agent (A2A) protocol specification. https://google.github.io/A2A/, 2024.
9. Hildebrandt, M., Li, H., Koner, R., et al. Scene graph reasoning for visual question answering. arXiv:2007.01072, 2020.
10. Hollmann, N., Mueller, S., & Hutter, F. Large language models for automated machine learning. arXiv:2402.00878, 2024.
11. Hong, S., Wang, X., Yu, J., et al. OpenDevin: An open platform for AI software developers as generalist agents. arXiv:2407.16741, 2024.
12. Hudson, D. & Manning, C. GQA: A new dataset for real-world visual reasoning and compositional question answering. CVPR, 2019.
13. Jimenez, C., Yang, J., Wettig, A., et al. SWE-Bench: Can language models resolve real-world GitHub issues? ICLR, 2024.
14. Jin, H., Song, Q., & Hu, X. AutoKeras: An AutoML library for deep learning. JMLR, 24(6):1--6, 2023.
15. Krishna, R., Zhu, Y., Groth, O., et al. Visual Genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 123:32--73, 2017.
16. Li, Y., Du, Y., Zhou, K., et al. Evaluating object hallucination in large vision-language models. EMNLP, 2023.
17. BerriAI. LiteLLM: Call 100+ LLM APIs using the OpenAI format. https://github.com/BerriAI/litellm, 2024.
18. Liu, H., Li, C., Wu, Q., & Lee, Y. Visual instruction tuning. NeurIPS, 2024.
19. OpenAI. GPT-4 technical report. arXiv:2303.08774, 2023.
20. Settles, B. Active learning literature survey. Computer Sciences Technical Report 1648, University of Wisconsin-Madison, 2009.
21. SignificantGravitas. AutoGPT: An autonomous GPT-4 experiment. https://github.com/Significant-Gravitas/AutoGPT, 2023.
22. Wang, L., Ma, C., Feng, X., et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):1--26, 2024.
23. Xiao, B., Wu, H., Xu, W., et al. Florence-2: Advancing a unified representation for a variety of vision tasks. CVPR, 2024.
24. Xie, S., Levy, O., et al. Active prompting with chain-of-thought for large language models. arXiv:2302.12246, 2024.
25. Xu, D., Zhu, Y., Choy, C., & Fei-Fei, L. Scene graph generation by iterative message passing. CVPR, 2017.
26. Yang, J., Jimenez, C., Wettig, A., et al. SWE-Agent: Agent-computer interfaces enable automated software engineering. arXiv:2405.15793, 2024.
27. Zhang, Y., Mao, H., Zheng, Y., et al. MLE-Agent: Automated machine learning engineering with LLM agents. arXiv:2402.15642, 2024.
