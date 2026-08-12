# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for collecting per-step training efficiency metrics."""

import logging
import math


_UTILIZATION_WARNING_EMITTED = False


def calculate_mfu(throughput_tflops, theoretical_device_tflops):
    """Calculate model FLOPs utilization as a percentage for one device."""
    if theoretical_device_tflops <= 0:
        raise ValueError("theoretical_device_tflops must be greater than 0")
    return throughput_tflops / theoretical_device_tflops * 100.0


def get_ai_core_utilization(torch_module=None, torch_npu_module=None):
    """Return mean AI Core utilization across ranks, ignoring unsupported samples.

    ``torch_npu.npu.utilization`` reports a device's combined Cube and Vector
    utilization. Each rank samples its current NPU, then the valid sums and counts
    are reduced so the value represents the whole training job.
    """
    if torch_module is None:
        import torch as torch_module
    if torch_npu_module is None:
        import torch_npu as torch_npu_module

    local_utilization = 0.0
    valid_sample = 0.0
    try:
        local_utilization = float(torch_npu_module.npu.utilization())
        if math.isfinite(local_utilization) and 0.0 <= local_utilization <= 100.0:
            valid_sample = 1.0
        else:
            local_utilization = 0.0
    except Exception as exc:  # Some driver/CANN combinations do not expose this counter.
        global _UTILIZATION_WARNING_EMITTED
        if not _UTILIZATION_WARNING_EMITTED:
            logging.warning("Unable to query NPU AI Core utilization: %s", exc)
            _UTILIZATION_WARNING_EMITTED = True

    device = 'cuda'
    if hasattr(torch_module, 'npu'):
        device = torch_module.npu.current_device()
    utilization_stats = torch_module.tensor(
        [local_utilization, valid_sample], dtype=torch_module.float32, device=device
    )

    distributed = getattr(torch_module, 'distributed', None)
    if (distributed is not None and distributed.is_available()
            and distributed.is_initialized()):
        distributed.all_reduce(utilization_stats)

    valid_samples = utilization_stats[1].item()
    if valid_samples == 0:
        return None
    return utilization_stats[0].item() / valid_samples
