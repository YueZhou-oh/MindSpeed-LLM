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
import threading


_UTILIZATION_WARNING_EMITTED = False


def calculate_mfu(throughput_tflops, theoretical_device_tflops):
    """Calculate model FLOPs utilization as a percentage for one device."""
    if theoretical_device_tflops <= 0:
        raise ValueError("theoretical_device_tflops must be greater than 0")
    return throughput_tflops / theoretical_device_tflops * 100.0


def _warn_utilization_query_failure(exc):
    global _UTILIZATION_WARNING_EMITTED
    if not _UTILIZATION_WARNING_EMITTED:
        logging.warning("Unable to query NPU AI Core utilization: %s", exc)
        _UTILIZATION_WARNING_EMITTED = True


class AiCoreUtilizationSampler:
    """Periodically sample one rank's NPU during a training iteration."""

    def __init__(self, sampling_interval=0.2, torch_npu_module=None, device_id=None):
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be greater than 0")

        if torch_npu_module is None:
            import torch_npu as torch_npu_module

        self.sampling_interval = sampling_interval
        self.torch_npu = torch_npu_module
        self.device_id = (
            self.torch_npu.npu.current_device() if device_id is None else device_id
        )
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._stopped = False
        self._active = False
        self._generation = 0
        self._sample_sum = 0.0
        self._sample_count = 0
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"ai-core-utilization-npu-{self.device_id}",
            daemon=True,
        )
        self._thread.start()

    def begin_interval(self):
        """Begin a new sampling window, discarding samples from the prior one."""
        with self._condition:
            self._generation += 1
            self._sample_sum = 0.0
            self._sample_count = 0
            self._active = True
            self._condition.notify_all()

    def end_interval(self):
        """Stop sampling and return ``(sample_sum, sample_count)``."""
        with self._condition:
            self._active = False
            self._condition.notify_all()
            return self._sample_sum, self._sample_count

    def close(self):
        """Stop the sampler thread."""
        with self._condition:
            self._stopped = True
            self._active = False
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.sampling_interval * 2))

    def _sample_loop(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopped or self._active)
                if self._stopped:
                    return
                generation = self._generation

            try:
                utilization = float(self.torch_npu.npu.utilization(self.device_id))
            except Exception as exc:  # Driver/CANN may not expose this counter.
                _warn_utilization_query_failure(exc)
            else:
                if math.isfinite(utilization) and 0.0 <= utilization <= 100.0:
                    with self._condition:
                        if self._active and generation == self._generation:
                            self._sample_sum += utilization
                            self._sample_count += 1

            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._stopped
                        or not self._active
                        or self._generation != generation
                    ),
                    timeout=self.sampling_interval,
                )


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
        _warn_utilization_query_failure(exc)

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


def reduce_ai_core_utilization(sample_sum, sample_count, torch_module=None):
    """Return the sample-weighted mean utilization across all training ranks."""
    if torch_module is None:
        import torch as torch_module

    device = 'cuda'
    if hasattr(torch_module, 'npu'):
        device = torch_module.npu.current_device()
    utilization_stats = torch_module.tensor(
        [sample_sum, sample_count], dtype=torch_module.float32, device=device
    )

    distributed = getattr(torch_module, 'distributed', None)
    if (distributed is not None and distributed.is_available()
            and distributed.is_initialized()):
        distributed.all_reduce(utilization_stats)

    total_samples = utilization_stats[1].item()
    if total_samples == 0:
        return None
    return utilization_stats[0].item() / total_samples
