# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Adapter from FSDP2 Qwen3 experts to HyperParallel's parameterless MegaMoE."""

from torch.distributed.tensor import DTensor


def get_experts_forward_megamoe_fn(ep_group):
    try:
        from hyper_parallel.core.multicore import MegaMoeExperts
    except ImportError as exc:
        raise ImportError(
            "ep_dispatcher=megamoe requires a compatible HyperParallel build with MegaMoeExperts."
        ) from exc

    executor = None

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        nonlocal executor
        hidden_size = self.hidden_dim
        intermediate_size = self.intermediate_size
        local_experts = self.num_local_experts

        # FSDP2 owns and materializes the parameters. Pass the current views on
        # every call so parameter refresh, gradients, and checkpoint keys stay in FSDP2.
        gate_up = self.gate_up_proj
        down = self.down_proj
        if isinstance(gate_up, DTensor):
            gate_up = gate_up.to_local()
        if isinstance(down, DTensor):
            down = down.to_local()
        weights = (
            gate_up.reshape(local_experts, hidden_size, 2 * intermediate_size),
            down.reshape(local_experts, intermediate_size, hidden_size),
        )
        if executor is None:
            executor = MegaMoeExperts(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=self.num_global_experts,
                top_k=top_k_index.shape[-1],
                ep_group=ep_group,
                create_parameters=False,
                expert_weight_layout="EIO",
            )
        return executor(hidden_states, top_k_index, top_k_weights, expert_weights=weights)

    return experts_forward
