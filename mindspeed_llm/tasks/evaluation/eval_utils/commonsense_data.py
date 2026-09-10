"""Harness-style zero-shot multiple-choice evaluation.
Task definitions: EleutherAI/lm-evaluation-harness (MIT),
revision b954108c9baaaa934b4ad842033b31a97ee30816.
"""
import json
import math
import re
from pathlib import Path

TASKS = {
    'piqa': ('baber/piqa', None, 'validation'),
    'social_iqa': ('lighteval/siqa', None, 'validation'),
    'hellaswag': ('Rowan/hellaswag', None, 'validation'),
    'arc_easy': ('allenai/ai2_arc', 'ARC-Easy', 'test'),
    'arc_challenge': ('allenai/ai2_arc', 'ARC-Challenge', 'test'),
    'openbookqa': ('allenai/openbookqa', 'main', 'test'),
    'winogrande': ('allenai/winogrande', 'winogrande_xl', 'validation'),
}
ALIASES = {'siqa': 'social_iqa', 'arc-e': 'arc_easy', 'arc-c': 'arc_challenge', 'obqa': 'openbookqa'}


def canonical_task(name):
    name = name.lower()
    return ALIASES.get(name, name)


def preprocess(text):
    return re.sub(r'\[.*?\]', '', text.strip().replace(' [title]', '. ')).replace('  ', ' ')


def format_example(task, doc):
    """Return harness context, unprefixed choices and zero-based target."""
    task = canonical_task(task)
    if task == 'piqa':
        context = 'Question: ' + doc['goal'] + '\nAnswer:'
        choices, gold = [doc['sol1'], doc['sol2']], int(doc['label'])
    elif task == 'social_iqa':
        context = 'Q: ' + doc['context'] + ' ' + doc['question'] + '\nA:'
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        gold = int(doc['label']) - 1
    elif task == 'hellaswag':
        if 'query' in doc and 'choices' in doc:
            context, choices = doc['query'], doc['choices']
            gold = int(doc['gold'] if 'gold' in doc else doc['label'])
        else:
            context = preprocess(doc['activity_label'] + ': ' + doc['ctx_a'] + ' ' + doc['ctx_b'].capitalize())
            choices, gold = [preprocess(x) for x in doc['endings']], int(doc['label'])
    elif task in ('arc_easy', 'arc_challenge', 'openbookqa'):
        context = doc['question_stem'] if task == 'openbookqa' else 'Question: ' + doc['question'] + '\nAnswer:'
        choices = doc['choices']['text']
        key = str(doc['answerKey'])
        if task == 'openbookqa':
            key = key.lstrip()
        if task in ("arc_easy", "arc_challenge"):
            # Only use this mapping for ARC files confirmed to store
            # answerKey as a zero-based choice index.
            gold = int(key)
            if not 0 <= gold < len(choices):
                raise ValueError(
                    f"Invalid zero-based answerKey={key!r} "
                    f"for {len(choices)} choices"
                )
        else:
            # Preserve OpenBookQA's label-based mapping.
            gold = [str(x) for x in doc["choices"]["label"]].index(key)
    elif task == 'winogrande':
        sentence = doc['sentence']

        if sentence.count('_') != 1:
            raise ValueError(
                f'Expected exactly one blank in WinoGrande: {sentence!r}'
            )

        prefix, suffix = sentence.split('_', 1)
        suffix = suffix.strip()

        options = [doc['option1'], doc['option2']]
        if any(not isinstance(option, str) or not option for option in options):
            raise ValueError('WinoGrande options must be nonempty strings')

        answer = str(doc['answer']).strip()
        if answer not in ('1', '2'):
            raise ValueError(
                f'Invalid WinoGrande answer {answer!r}; '
                'use the labeled validation split'
            )

        # Each option provides a different conditioning context.
        context = [prefix + option for option in options]

        # Score the same sentence suffix under both contexts.
        choices = [suffix, suffix]
        gold = int(answer) - 1
    else:
        raise ValueError(f'Unknown task: {task}')
    if not context or not choices or any(not isinstance(x, str) or not x for x in choices):
        raise ValueError('Context and choices must be nonempty strings')
    if not 0 <= gold < len(choices):
        raise ValueError(f'Invalid gold label {gold}; use the labeled evaluation split')
    return context, choices, gold


def encode_pair(tokenizer, context, continuation):
    # Harness moves trailing whitespace to the continuation before tokenizing.
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    prefix = tokenizer.encode(context, add_special_tokens=False)
    joined = tokenizer.encode(context + continuation, add_special_tokens=False)
    target = joined[len(prefix):]
    if not prefix or not target:
        raise ValueError('Tokenized context and continuation must both be nonempty')
    return prefix + target, len(prefix)


def predictions(scores, choices):
    if len(scores) != len(choices) or not all(math.isfinite(x) for x in scores):
        raise ValueError('Expected one finite log-likelihood per choice')
    return (max(range(len(scores)), key=lambda i: scores[i]),
            max(range(len(scores)), key=lambda i: scores[i] / len(choices[i])))


def load_documents(path, task):
    """Load an explicit JSONL/JSON/parquet file or save_to_disk dataset."""
    path = Path(path)
    if path.is_dir():
        from datasets import load_from_disk, DatasetDict
        data = load_from_disk(str(path))
        return data[TASKS[task][2]] if isinstance(data, DatasetDict) else data
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if path.suffix == '.json':
        with path.open(encoding='utf-8') as stream:
            data = json.load(stream)
        if not isinstance(data, list):
            raise ValueError('JSON input must be a list of examples')
        return data
    if path.suffix == '.parquet':
        from datasets import load_dataset
        return load_dataset('parquet', data_files=str(path), split='train')
    raise ValueError(f'Unsupported dataset path: {path}')

