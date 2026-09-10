"""Teacher-forced scoring through the MCore model forward (PP=CP=1)."""
import torch
import torch.nn.functional as F
from megatron.training import get_args
from mindspeed_llm.tasks.evaluation.eval_impl.commonsense_eval import encode_pair


def loglikelihood(chat, requests):
    args = get_args()
    if (args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1
            or getattr(args, 'sequence_parallel', False)):
        raise ValueError('Commonsense scoring requires PP=1, CP=1 and sequence parallel disabled; TP is supported')
    if not args.use_mcore_models:
        raise ValueError('Commonsense scoring currently requires --use-mcore-models')
    chat.model.eval()
    results = []
    # One candidate per forward avoids padding and mask ambiguity.
    for context, continuation in requests:
        ids, boundary = encode_pair(chat.tokenizer, context, continuation)
        if len(ids) > args.max_position_embeddings:
            raise ValueError(f'Example has {len(ids)} tokens, exceeding max-position-embeddings; no silent truncation')
        tokens = torch.tensor([ids], dtype=torch.long, device=torch.cuda.current_device())
        with torch.inference_mode():
            seq_len = tokens.shape[1]
            position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
            # Megatron boolean masks use True for positions that are blocked.
            attention_mask = torch.ones(
                (1, 1, seq_len, seq_len), dtype=torch.bool, device=tokens.device
            ).triu(diagonal=1)
            logits = chat.model(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            if logits.ndim != 3 or logits.shape[:2] != tokens.shape:
                raise ValueError(f'Expected [batch, sequence, vocabulary] logits, got {logits.shape}')
            if logits.shape[-1] < len(chat.tokenizer):
                raise ValueError('Expected gathered full-vocabulary logits')
            selected = logits[0, boundary - 1:len(ids) - 1, :].float()
            targets = tokens[0, boundary:]
            score = -F.cross_entropy(selected, targets, reduction='sum')
        results.append(score.item())
    return results
