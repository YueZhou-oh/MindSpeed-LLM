#!/usr/bin/env python3

import os
import numpy as np
import torch

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


# Every path is the preprocessing --output-prefix,
# without "_packed_input_ids_document".
ROOT = "/dpc-zhouy/zhouy/instructft_data/FLAN_4K"

SOURCES = [

    ("cot_zsopt",    0.02, f"{ROOT}/cot_zsopt_data"),
    ("cot_fsopt",    0.04, f"{ROOT}/cot_fsopt_data"),

    ("dialog_zsopt", 0.6, f"{ROOT}/dialog_zsopt_data"),
    ("dialog_fsopt", 0.34, f"{ROOT}/dialog_fsopt_data"),
]


# SOURCES = [
#     # FLAN 2021: 40%
#     ("flan_zsopt",    0.10, f"{ROOT}/flan_zsopt_data"),
#     ("flan_zsnoopt",  0.10, f"{ROOT}/flan_zsnoopt_data"),
#     ("flan_fsopt",    0.10, f"{ROOT}/flan_fsopt_data"),
#     ("flan_fsnoopt",  0.10, f"{ROOT}/flan_fsnoopt_data"),

#     # T0: 32%
#     ("t0_zsopt",      0.08, f"{ROOT}/t0_zsopt_data"),
#     ("t0_zsnoopt",    0.08, f"{ROOT}/t0_zsnoopt_data"),
#     ("t0_fsopt",      0.08, f"{ROOT}/t0_fsopt_data"),
#     ("t0_fsnoopt",    0.08, f"{ROOT}/t0_fsnoopt_data"),

#     # NIv2: 20%
#     ("niv2_zsopt",    0.10, f"{ROOT}/niv2_zsopt_data"),
#     ("niv2_fsopt",    0.10, f"{ROOT}/niv2_fsopt_data"),

#     # CoT: 5%
#     ("cot_zsopt",    0.025, f"{ROOT}/cot_zsopt_data"),
#     ("cot_fsopt",    0.025, f"{ROOT}/cot_fsopt_data"),

#     # Dialog: 3%
#     ("dialog_zsopt", 0.015, f"{ROOT}/dialog_zsopt_data"),
#     ("dialog_fsopt", 0.015, f"{ROOT}/dialog_fsopt_data"),
# ]

OUTPUT_PREFIX = f"{ROOT}/flan2022_mix_4K_5m_demo"

# Start with 2M–5M examples rather than the whole 296+ GB collection.
TOTAL_SAMPLES = 5_000_000
SEED = 42

KEYS = [
    "input_ids",
    "attention_mask",
    "labels",
]


def indexed_prefix(base, key):
    return f"{base}_packed_{key}_document"


def main():
    if not np.isclose(sum(weight for _, weight, _ in SOURCES), 1.0):
        raise ValueError("Mixture weights must sum to 1.0")

    os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)

    # Refuse to overwrite an existing mixture.
    for key in KEYS:
        prefix = indexed_prefix(OUTPUT_PREFIX, key)
        if os.path.exists(get_bin_path(prefix)) or os.path.exists(get_idx_path(prefix)):
            raise FileExistsError(f"Output already exists: {prefix}")

    datasets = []
    counts = []

    for name, weight, base in SOURCES:
        group = {
            key: IndexedDataset(indexed_prefix(base, key), multimodal=False)
            for key in KEYS
        }

        lengths = {key: len(dataset) for key, dataset in group.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{name}: unaligned datasets: {lengths}")

        count = int(round(TOTAL_SAMPLES * weight))
        datasets.append(group)
        counts.append(count)

        print(
            f"{name:18s} available={next(iter(lengths.values())):,} "
            f"selected={count:,}"
        )

    # Correct any rounding difference.
    counts[-1] += TOTAL_SAMPLES - sum(counts)

    # Preserve the dtype of each indexed field.
    builders = {}
    for key in KEYS:
        reference_dtype = datasets[0][key].index.dtype

        for source_id, group in enumerate(datasets):
            if group[key].index.dtype != reference_dtype:
                raise TypeError(
                    f"{KEYS[source_id]} has inconsistent dtype for {key}"
                )

        output = indexed_prefix(OUTPUT_PREFIX, key)
        builders[key] = IndexedDatasetBuilder(
            get_bin_path(output),
            dtype=reference_dtype,
            multimodal=False,
        )

    rng = np.random.default_rng(SEED)

    # Select document IDs independently from each source. Small datasets are
    # sampled with replacement when the requested count exceeds their size.
    selected_indices = []

    for source_id, group in enumerate(datasets):
        available = len(group["input_ids"])
        requested = counts[source_id]
        replace = requested > available

        indices = rng.choice(
            available,
            size=requested,
            replace=replace,
        )
        selected_indices.append(indices)

        if replace:
            print(
                f"WARNING: {SOURCES[source_id][0]} is sampled with replacement"
            )

    # Globally interleave sources. This is important because MindSpeed performs
    # train/validation/test splitting on contiguous document ranges.
    source_schedule = np.concatenate([
        np.full(count, source_id, dtype=np.uint8)
        for source_id, count in enumerate(counts)
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
            print(f"Written {output_id + 1:,}/{TOTAL_SAMPLES:,}")

    for key, builder in builders.items():
        output = indexed_prefix(OUTPUT_PREFIX, key)
        builder.finalize(get_idx_path(output))

    print(f"Finished: {OUTPUT_PREFIX}")
    print(f'Training DATA_PATH="{OUTPUT_PREFIX}"')


if __name__ == "__main__":
    main()
