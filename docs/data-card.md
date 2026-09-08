# Data card

The project uses the Kaggle LLM Classification Finetuning competition data,
which is based on human preferences between chatbot responses. The competition
expects a probability for each of `winner_model_a`, `winner_model_b`, and
`winner_tie`; submissions are evaluated with multiclass log loss.

Expected inputs:

- `prompt`
- `response_a`
- `response_b`

Training rows additionally contain one-hot target probabilities. This public
repository does not redistribute the raw competition data. Download it from
[Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning)
under the competition terms and validate it with the package utilities.

The Kaggle data page identifies the dataset license as CC BY-NC 4.0. Readers
must verify the current terms before using the data outside the competition.
