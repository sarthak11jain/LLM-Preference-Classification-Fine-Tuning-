# Phase 2 — Repository design decisions

Phase 2 finalizes the public repository contract. It does not yet perform the
full source migration or model refactor; those belong to Phase 3 onward.

## Identity

- Repository name: `LLM-Preference-Classification-Fine-Tuning-`
- Intended visibility: public
- Default branch: `main`
- Code license: MIT, subject to confirmation against any reused source
- Competition data: external download only; no raw data committed
- Model weights/adapters: external/private; no large binaries committed

## Public information architecture

```text
LLM-Preference-Classification-Fine-Tuning-/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .github/workflows/quality.yml
├── configs/
│   ├── modernbert_lora.yaml
│   └── gemma2_lora.example.yaml
├── docs/
│   ├── architecture.md
│   ├── data-card.md
│   ├── external-sources.md
│   ├── kaggle.md
│   ├── reproducibility.md
│   ├── cv-provenance.md
│   ├── phase-1-evidence-audit.md
│   ├── phase-2-repository-design.md
│   └── kaggle-submission-audit.md
├── evidence/
│   └── small metric/provenance artifacts only
├── experiments/
│   ├── results.csv
│   ├── artifact-inventory.csv
│   └── kaggle-submission-audit.csv
├── notebooks/
│   └── selected Gemma and ModernBERT training/inference records
├── src/preference_classifier/
│   └── canonical reusable implementation
└── tests/
    └── deterministic and smoke tests
```

## Modeling scope

Gemma-2 9B and ModernBERT-large are co-primary workflows because both are
explicitly represented in the CV. The repository will not describe Gemma-2 as
a side experiment. Their differences will be documented honestly:

| Workflow | Role | Compute expectation | Current evidence state |
|---|---|---|---|
| Gemma-2 9B LoRA | Primary decoder workflow | High-memory external GPU/Kaggle GPU | Fold-1 local metric artifact exists |
| ModernBERT-large LoRA | Primary encoder workflow | GPU required for practical training | Implementation and recorded fold value exist; exact metric artifact needs recovery |
| ModernBERT frozen head | Baseline | GPU/CPU depending on model access | Local smoke log exists; claimed public score needs Kaggle verification |
| TF-IDF/logistic regression | Lightweight text baseline | CPU | Existing source implementation and local result recorded in original README |

## Dependency policy

The public package will use `pyproject.toml` as the canonical dependency file:

- base install: NumPy, pandas, and scikit-learn for schema, baselines, and metrics;
- `[dev]`: pytest and linting tools;
- `[training]`: PyTorch, Transformers, PEFT, and Accelerate;
- Kaggle and Vertex tooling: optional operational dependencies, not required for the base package;
- no downloaded wheels or model caches in Git.

Training code must import heavy dependencies lazily where practical so the
baseline and tests remain usable on a CPU-only machine.

## Canonical implementation boundary

`src/preference_classifier/` is the canonical source for reusable logic. The
notebooks remain experiment records and Kaggle templates; they must not become
the only place where a CV claim is implemented. Later phases will migrate and
deduplicate the original monolithic scripts into this package.

## Results presentation policy

The README will show all relevant results, but every row will include:

- model and training mode;
- metric and exact value;
- local fold, OOF, public leaderboard, or external scope;
- evidence path or URL;
- verification status.

The project can retain the CV numbers while distinguishing “recorded from the
original project” from “independently reproduced in the public repository.”

## Design assets planned

The final presentation will contain:

1. a system overview diagram showing both model paths;
2. a response-swap/TTA diagram showing probability remapping;
3. a results chart separating local validation and public leaderboard scores;
4. a compact repository tree in the README.

These assets are presentation work for a later phase, not new experimental
evidence.
