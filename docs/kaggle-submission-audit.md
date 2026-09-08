# Kaggle submission audit

The local source project contains seven Kaggle submission packages. Their
metadata and local output availability are summarized in
[`experiments/kaggle-submission-audit.csv`](../experiments/kaggle-submission-audit.csv).

The audit found:

- the CLI returned 18 account kernels for this competition and 11 competition
  submission records;
- `try-sj-12` has a machine-readable ModernBERT fold-1 metric artifact with
  `1.0121814648626144`;
- `try-sj-16` has a machine-readable three-fold ModernBERT inference artifact
  and the following public submission scored `1.01414`;
- `try-sj-13` is the fold-1 ModernBERT inference workflow and the following
  public submission scored `1.01474`;
- `try-sj-11` is the ModernBERT frozen-head baseline and the following public
  submission scored `1.07855`;
- `submission_023` is a separate smoke run with 300 training rows, a
  256-token limit, and local validation log loss `1.14524`;
- the Gemma `0.99655` value is supported separately by the saved fold metrics
  artifact under `evidence/`; the Gemma inference submissions currently have
  blank public scores in the CLI submission history.

The authenticated Kaggle CLI was used to retrieve kernel metadata, source
notebooks, output files, execution logs, and the competition submission history.
The browser remains a backup for inspecting pages that the CLI cannot expose.

The public README may now present the CLI-backed ModernBERT scores as verified,
while continuing to label Gemma `0.99655` as a local fold-1 validation result.
