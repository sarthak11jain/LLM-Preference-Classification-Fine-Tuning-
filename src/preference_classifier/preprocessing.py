"""Text formatting and token-budget helpers shared by model workflows."""

from collections.abc import Sequence


def format_comparison(prompt: str, response_a: str, response_b: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}"


def build_text_inputs(prompt: str, response_a: str, response_b: str, swap_responses: bool = False) -> str:
    if swap_responses:
        response_a, response_b = response_b, response_a
    return format_comparison(prompt, response_a, response_b)


def truncate_head_tail(tokens: Sequence[int], budget: int, head_ratio: float = 0.5) -> list[int]:
    if budget < 0 or not 0 <= head_ratio <= 1:
        raise ValueError("budget must be non-negative and head_ratio must be in [0, 1]")
    tokens = list(tokens)
    if len(tokens) <= budget:
        return tokens
    head = int(budget * head_ratio)
    return tokens[:head] + tokens[-(budget - head) :]


def allocate_budgets(lengths: Sequence[int], total_budget: int, shares: Sequence[float]) -> list[int]:
    if len(lengths) != len(shares) or not lengths:
        raise ValueError("lengths and shares must be non-empty and have equal length")
    if total_budget < 0 or any(length < 0 for length in lengths):
        raise ValueError("lengths and total_budget must be non-negative")
    if any(share < 0 for share in shares) or sum(shares) <= 0:
        raise ValueError("shares must be non-negative and have a positive sum")
    total_share = sum(shares)
    raw = [total_budget * share / total_share for share in shares]
    budgets = [min(length, int(value)) for length, value in zip(lengths, raw)]
    remaining = total_budget - sum(budgets)
    while remaining > 0:
        candidates = [i for i, length in enumerate(lengths) if budgets[i] < length]
        if not candidates:
            break
        index = max(candidates, key=lambda i: raw[i] - budgets[i])
        budgets[index] += 1
        remaining -= 1
    return budgets
