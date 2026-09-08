# Architecture

The system is a three-class sequence classifier. The prompt and both candidate
responses are formatted into one model input. A Gemma-2 9B decoder or a
ModernBERT-large encoder receives the sequence, while LoRA keeps the trainable
update small; a three-way classification head produces logits and softmax
probabilities. Both model families are first-class workflows rather than a
primary model plus an unrelated experiment.

Training uses prompt groups for fold assignment so repeated prompts do not leak
across validation boundaries. Each training row is duplicated with responses A
and B exchanged, and the A/B labels are exchanged with them. At inference, the
swapped prediction is permuted back before averaging with the original prediction.
