# Experiment records

`results.csv` contains only metrics that can be traced to the original project
artifacts. Values are labeled by evaluation scope. Local validation is not a
leaderboard score, and the Gemma-2 result is fold 1 rather than a completed
three-fold aggregate.

New runs should record the model revision, seed, fold rule, data version, token
length, label smoothing, hardware, and output artifact together with the metric.
