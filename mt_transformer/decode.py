from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import TransformerTranslator
from .tokenizer import BPETokenizer


@torch.no_grad()
def greedy_translate(
    model: TransformerTranslator,
    tokenizer: BPETokenizer,
    sentence: str,
    device: torch.device,
    max_len: int = 80,
) -> str:
    model.eval()
    src = torch.tensor([tokenizer.encode(sentence, max_len=max_len)], dtype=torch.long, device=device)
    memory, src_padding = model.encode(src)
    ys = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        logits = model.decode_step(ys, memory, src_padding)
        next_id = int(logits.argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ys.squeeze(0).tolist())


@torch.no_grad()
def beam_translate(
    model: TransformerTranslator,
    tokenizer: BPETokenizer,
    sentence: str,
    device: torch.device,
    beam_size: int = 4,
    max_len: int = 80,
    length_penalty: float = 0.7,
) -> str:
    model.eval()
    src = torch.tensor([tokenizer.encode(sentence, max_len=max_len)], dtype=torch.long, device=device)
    memory, src_padding = model.encode(src)
    beams: list[tuple[list[int], float, bool]] = [([tokenizer.bos_id], 0.0, False)]

    for _ in range(max_len - 1):
        candidates: list[tuple[list[int], float, bool]] = []
        for ids, score, finished in beams:
            if finished:
                candidates.append((ids, score, True))
                continue
            ys = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model.decode_step(ys, memory, src_padding)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            top_scores, top_ids = torch.topk(log_probs, beam_size)
            for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                next_ids = ids + [int(token_id)]
                candidates.append((next_ids, score + float(token_score), token_id == tokenizer.eos_id))
        beams = sorted(candidates, key=lambda x: _normalized_score(x[1], len(x[0]), length_penalty), reverse=True)[
            :beam_size
        ]
        if all(done for _, _, done in beams):
            break

    best_ids = max(beams, key=lambda x: _normalized_score(x[1], len(x[0]), length_penalty))[0]
    return tokenizer.decode(best_ids)


def _normalized_score(score: float, length: int, penalty: float) -> float:
    return score / (((5.0 + length) / 6.0) ** penalty)
