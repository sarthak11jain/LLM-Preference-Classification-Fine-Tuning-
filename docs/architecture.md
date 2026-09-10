# Architecture

The system is a three-class sequence classifier. The prompt and both candidate
responses are formatted into one model input. A Gemma-2 9B decoder or a
ModernBERT-large encoder receives the sequence, while LoRA keeps the trainable
update small; a three-way classification head produces logits and softmax
probabilities. Both model families are first-class workflows rather than a
primary model plus an unrelated experiment.

The package supports two explicit validation tracks. The primary Gemma result
uses the deterministic `id % 5 == 0` split; the leakage-aware comparison track
uses prompt groups so repeated prompts do not cross fold boundaries. Each
training row can be duplicated with responses A and B exchanged, and the A/B
labels exchanged with them. At inference, the swapped prediction is permuted
back before averaging with the original prediction.
