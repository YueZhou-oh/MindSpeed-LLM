#!/usr/bin/env python3

"""Build an approximately 20B-token final SFT mixture.

`fraction` is the fraction retained from that source, not its weight in the
final mixture. `start` controls where sampling begins:

* an integer is an absolute document index, e.g. ``start=100_000``;
* a float in [0, 1) is a relative point, e.g. ``start=0.25`` starts at 25%.

Each source is sampled as a circular window. If the requested window reaches
the end of a dataset, it continues from document 0. The selected document IDs
are shuffled before all sources are globally interleaved. The same IDs are
always applied to input_ids, attention_mask, and labels.
"""

import math
import os
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

import sys
sys.path.append("/dpc-zhouy/zhouy/MindSpeed-LLM")
from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


DATA_ROOT = "/dpc-zhouy/zhouy/instructft_data"


@dataclass(frozen=True)
class Source:
    name: str
    fraction: float
    start: Union[int, float]
    base: str
    original_tokens: int


# Fractions implement a capacity-aware approximately 20B-token recipe. Small
# sources are capped at one complete pass; their unavailable allocation is
# redistributed proportionally across sources that still have unused data.
# Fractions are based on original token counts and applied to document counts,
# so the final token count is approximate when document lengths are non-uniform.
#
# Change `start` independently for any source. Examples:
#   start=500_000  -> begin at absolute document index 500,000
#   start=0.25     -> begin at the 25% point of that source

# SOURCES = [
#     # FLAN family: 25% of the final mixture. The pre-built 5M FLAN mixture is
#     # deliberately excluded.
#     Source("cot_fsopt",       1.000000000, 0, f"{DATA_ROOT}/FLAN_4K/cot_fsopt_data",       85_553_325),
#     Source("cot_zsopt",       1.000000000, 0, f"{DATA_ROOT}/FLAN_4K/cot_zsopt_data",       10_999_763),
#     Source("dialog_fsopt",    0.016127907, 0, f"{DATA_ROOT}/FLAN_4K/dialog_fsopt_data", 3_911_493_247),
#     Source("dialog_zsopt",    0.326773023, 0, f"{DATA_ROOT}/FLAN_4K/dialog_zsopt_data",   450_454_761),
#     Source("flan_fsnoopt",    0.041893846, 0, f"{DATA_ROOT}/FLAN_4K/flan_fsnoopt_data", 10_038_737_670),
#     Source("flan_fsopt",      0.017129978, 0, f"{DATA_ROOT}/FLAN_4K/flan_fsopt_data",   24_551_188_737),
#     Source("flan_zsnoopt",    0.234520682, 0, f"{DATA_ROOT}/FLAN_4K/flan_zsnoopt_data",  4_184_320_766),
#     Source("flan_zsopt",      0.218434002, 0, f"{DATA_ROOT}/FLAN_4K/flan_zsopt_data",    4_492_477_131),
#     Source("niv2_fsopt",      0.100827678, 0, f"{DATA_ROOT}/FLAN_4K/niv2_fsopt_data",    4_171_090_083),
#     Source("niv2_zsopt",      0.883584328, 0, f"{DATA_ROOT}/FLAN_4K/niv2_zsopt_data",    1_110_601_138),
#     Source("t0_fsnoopt",      0.019879355, 0, f"{DATA_ROOT}/FLAN_4K/t0_fsnoopt_data",   16_924_545_796),
#     Source("t0_fsopt",        0.007541506, 0, f"{DATA_ROOT}/FLAN_4K/t0_fsopt_data",     44_612_983_312),
#     Source("t0_zsnoopt",      0.114979946, 0, f"{DATA_ROOT}/FLAN_4K/t0_zsnoopt_data",    6_827_693_336),
#     Source("t0_zsopt",        0.107482749, 0, f"{DATA_ROOT}/FLAN_4K/t0_zsopt_data",      7_303_942_415),

#     # General CoT, chat, science, math, code, safety, and knowledge data.
#     Source("alpaca_cot",      0.028932761, 0, f"{DATA_ROOT}/alpaca_cot_4K/alpaca_cot_english_no_flan", 38_762_179_933),
#     Source("nemotron_chat",   1.000000000, 0, f"{DATA_ROOT}/nemotron_bin_4K/chat/chat",                 53_952_947),
#     Source("nemotron_code",   0.70, 0, f"{DATA_ROOT}/nemotron_bin_4K/code/code_v1.1",        1_855_900_337),
#     Source("nemotron_math",   0.20, 0, f"{DATA_ROOT}/nemotron_bin_4K/math/math_v1.1",        7_846_816_315),
#     Source("nemotron_safety", 1.000000, 0, f"{DATA_ROOT}/nemotron_bin_4K/safety/safety",             9_792_134),
#     Source("nemotron_science",1.000000000, 0, f"{DATA_ROOT}/nemotron_bin_4K/science/science",       1_247_599_518),
#     Source("fact_seeking",    1.000000000, 0, f"{DATA_ROOT}/nemotron_specialized_4K/fact_seeking/fact_seeking_sharegpt_00000", 89_707_993),
#     Source("ultrachat",       1.000000000, 0, f"{DATA_ROOT}/ultrachat/ultrachat_4K",                 1_857_242_283),
#     Source("ultradata_code",  0.50, 0, f"{DATA_ROOT}/ultrachat-sft/Code_4K",                  1_861_175_541),

#     # including chinese
#     # Source("ultradata_if",    1.000000000, 0, f"{DATA_ROOT}/ultrachat-sft/IF_4K",                       89_399_656),
#     Source("ultradata_knowledge", 1.000000000, 0, f"{DATA_ROOT}/ultrachat-sft/Knowledge_4K",          364_141_341),
#     Source("ultradata_math",  0.20, 0, f"{DATA_ROOT}/ultrachat-sft/Math_4K",                  5_810_460_902),
# ]

# OUTPUT_PREFIX = f"{DATA_ROOT}/final_sft_4K_16b_0922"
# TARGET_ESTIMATED_TOKENS = 16_000_000_000
# SEED = 42


SOURCES = [
    Source("ownership",       1.000000000, 0, f"{DATA_ROOT}/ownership",       300),
    Source("honeybee",        1.000000000, 0, f"{DATA_ROOT}/honeybee",       2_307),
    Source("dialog_fsopt",    0.001, 0.5, f"{DATA_ROOT}/FLAN_4K/dialog_fsopt_data", 3_911_493_247),
    Source("t0_fsnoopt",      0.0001, 0.5, f"{DATA_ROOT}/FLAN_4K/t0_fsnoopt_data",   16_924_545_796),
    Source("alpaca_cot",      0.0001, 0.5, f"{DATA_ROOT}/alpaca_cot_4K/alpaca_cot_english_no_flan", 38_762_179_933),

    Source("flan_zsnoopt",    0.001, 0.5, f"{DATA_ROOT}/FLAN_4K/flan_zsnoopt_data",  4_184_320_766),
    Source("flan_zsopt",      0.001, 0.5, f"{DATA_ROOT}/FLAN_4K/flan_zsopt_data",    4_492_477_131),

    Source("nemotron_code",   0.02, 0.8, f"{DATA_ROOT}/nemotron_bin_4K/code/code_v1.1",        1_855_900_337),
    Source("nemotron_math",   0.01, 0.5, f"{DATA_ROOT}/nemotron_bin_4K/math/math_v1.1",        7_846_816_315),
    Source("nemotron_science",0.02, 0.5, f"{DATA_ROOT}/nemotron_bin_4K/science/science",       1_247_599_518),
]

OUTPUT_PREFIX = f"{DATA_ROOT}/added_sft_4K_100k_1010"
TARGET_ESTIMATED_TOKENS = 100_000
SEED = 42

'''
ownership              available=        300 fraction= 1.000000 start=          0 selected=       300
honeybee               available=      2,307 fraction= 1.000000 start=          0 selected=     2,307
dialog_fsopt           available=  5,425,015 fraction= 0.001000 start=  2,712,507 selected=     5,425
t0_fsnoopt             available= 32,191,051 fraction= 0.000100 start= 16,095,525 selected=     3,219
alpaca_cot             available= 80,403,819 fraction= 0.000100 start= 40,201,909 selected=     8,040
flan_zsnoopt           available= 37,384,598 fraction= 0.001000 start= 18,692,299 selected=    37,385
flan_zsopt             available= 38,970,972 fraction= 0.001000 start= 19,485,486 selected=    38,971
nemotron_code          available=    496,206 fraction= 0.020000 start=    396,964 selected=     9,924
nemotron_math          available=  2,225,427 fraction= 0.010000 start=  1,112,713 selected=    22,254
nemotron_science       available=    708,920 fraction= 0.020000 start=    354,460 selected=    14,178
Total selected documents: 142,003
Estimated stored tokens: 158,697,731
'''

KEYS = [
    "input_ids",
    "attention_mask",
    "labels",
]


def indexed_prefix(base, key):
    return f"{base}_packed_{key}_document"


def resolve_start(start, available, name):
    """Convert an absolute index or relative point into a document index."""
    if isinstance(start, bool):
        raise TypeError(f"{name}: start must be an int or float, not bool")

    if isinstance(start, int):
        if not 0 <= start < available:
            raise ValueError(
                f"{name}: absolute start {start:,} is outside "
                f"[0, {available - 1:,}]"
            )
        return start

    if isinstance(start, float):
        if not 0.0 <= start < 1.0:
            raise ValueError(f"{name}: relative start must be in [0, 1)")
        return min(int(math.floor(start * available)), available - 1)

    raise TypeError(f"{name}: start must be an int or float")


def circular_indices(available, count, start_index):
    """Select `count` unique indices, wrapping once at the dataset boundary."""
    if not 0 <= count <= available:
        raise ValueError(
            f"count must be in [0, {available:,}], received {count:,}"
        )
    return (start_index + np.arange(count, dtype=np.int64)) % available


def main():
    if not SOURCES:
        raise ValueError("SOURCES cannot be empty")

    names = [source.name for source in SOURCES]
    if len(names) != len(set(names)):
        raise ValueError("Every source name must be unique")

    for source in SOURCES:
        if not 0.0 <= source.fraction <= 1.0:
            raise ValueError(
                f"{source.name}: fraction must be in [0, 1], "
                f"received {source.fraction}"
            )

    os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)

    # Refuse to overwrite an existing mixture.
    for key in KEYS:
        prefix = indexed_prefix(OUTPUT_PREFIX, key)
        if os.path.exists(get_bin_path(prefix)) or os.path.exists(get_idx_path(prefix)):
            raise FileExistsError(f"Output already exists: {prefix}")

    datasets = []
    counts = []
    selected_indices = []
    estimated_tokens = 0.0
    rng = np.random.default_rng(SEED)

    for source in SOURCES:
        group = {
            key: IndexedDataset(indexed_prefix(source.base, key), multimodal=False)
            for key in KEYS
        }

        lengths = {key: len(dataset) for key, dataset in group.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{source.name}: unaligned datasets: {lengths}")

        available = lengths["input_ids"]
        if available == 0:
            raise ValueError(f"{source.name}: source dataset is empty")

        count = int(round(available * source.fraction))
        start_index = resolve_start(source.start, available, source.name)
        indices = circular_indices(available, count, start_index)
        rng.shuffle(indices)

        datasets.append(group)
        counts.append(count)
        selected_indices.append(indices)
        estimated_tokens += source.original_tokens * source.fraction

        print(
            f"{source.name:22s} available={available:>11,} "
            f"fraction={source.fraction:>9.6f} "
            f"start={start_index:>11,} selected={count:>10,}"
        )

    total_samples = sum(counts)
    if total_samples == 0:
        raise ValueError("The configured fractions select zero documents")

    print(f"Total selected documents: {total_samples:,}")
    print(f"Estimated stored tokens: {estimated_tokens:,.0f}")
    # if not math.isclose(
    #     estimated_tokens,
    #     TARGET_ESTIMATED_TOKENS,
    #     rel_tol=1e-4,
    # ):
    #     raise ValueError(
    #         f"Estimated mixture has {estimated_tokens:,.0f} tokens; expected "
    #         f"approximately {TARGET_ESTIMATED_TOKENS:,}"
    #     )

    # Preserve the dtype of each indexed field.
    builders = {}
    for key in KEYS:
        reference_dtype = datasets[0][key].index.dtype

        for source, group in zip(SOURCES, datasets):
            if group[key].index.dtype != reference_dtype:
                raise TypeError(
                    f"{source.name}: inconsistent dtype for {key}: "
                    f"{group[key].index.dtype} != {reference_dtype}"
                )

        output = indexed_prefix(OUTPUT_PREFIX, key)
        builders[key] = IndexedDatasetBuilder(
            get_bin_path(output),
            dtype=reference_dtype,
            multimodal=False,
        )

    # Globally interleave sources. This is important because MindSpeed performs
    # train/validation/test splitting on contiguous document ranges.
    source_dtype = np.min_scalar_type(len(SOURCES) - 1)
    source_schedule = np.concatenate([
        np.full(count, source_id, dtype=source_dtype)
        for source_id, count in enumerate(counts)
        if count > 0
    ])
    rng.shuffle(source_schedule)

    source_positions = np.zeros(len(SOURCES), dtype=np.int64)

    for output_id, source_id_value in enumerate(source_schedule):
        source_id = int(source_id_value)
        position = source_positions[source_id]
        document_id = int(selected_indices[source_id][position])
        source_positions[source_id] += 1

        for key in KEYS:
            item = datasets[source_id][key][document_id]
            tensor = torch.from_numpy(np.asarray(item).copy())
            builders[key].add_item(tensor)
            builders[key].end_document()

        if (output_id + 1) % 100_000 == 0:
            print(f"Written {output_id + 1:,}/{total_samples:,}")

    for key, builder in builders.items():
        output = indexed_prefix(OUTPUT_PREFIX, key)
        builder.finalize(get_idx_path(output))

    print(f"Finished: {OUTPUT_PREFIX}")
    print(f'Training DATA_PATH="{OUTPUT_PREFIX}"')


if __name__ == "__main__":
    main()
