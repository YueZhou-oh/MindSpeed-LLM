# Copyright 2025 Moonshot AI and the LlamaFactory team.

import math
import torch

try:
    from torch.distributed.tensor import DTensor, Replicate
except ImportError:
    DTensor = None


# Five-step asymmetric schedule also used by HyperParallel and MindSpeed Core.
_ASYM5_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


def zeropower_via_newtonschulz5(G: "torch.Tensor", steps: int, variant: str = "legacy") -> "torch.Tensor":
    """Orthogonalize a matrix with legacy or asymmetric Newton-Schulz steps."""
    if variant == "legacy":
        coefficients = ((3.4445, -4.7750, 2.0315),) * steps
    elif variant == "asym5" and 1 <= steps <= len(_ASYM5_COEFFICIENTS):
        coefficients = _ASYM5_COEFFICIENTS[:steps]
    else:
        raise ValueError("Muon requires legacy, or asym5 with one to five Newton-Schulz steps.")

    distributed = DTensor is not None and isinstance(G, DTensor)
    original = G
    if distributed:
        # Asym5 uses the logical matrix rather than independently normalizing each shard.
        G = G.full_tensor() if variant == "asym5" else G.to_local()
    if G.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(G.shape)}.")
    # Convert to float32 to prevent NaN values
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for a, b, c in coefficients:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    if not distributed:
        return X
    if variant == "asym5":
        replicated = DTensor.from_local(
            X, original.device_mesh, tuple(Replicate() for _ in original.placements),
        )
        return replicated.redistribute(placements=original.placements)
    return DTensor.from_local(
        X, original.device_mesh, original.placements,
        shape=original.shape, stride=original.stride(),
    )


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations.
        ns_variant: Use the legacy local update or Asym5 on the full logical matrix.
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
            self,
            lr=1e-3,
            wd=0.1,
            muon_params=None,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_params=None,
            adamw_betas=(0.9, 0.95),
            adamw_eps=1e-8,
            ns_variant="legacy",
    ):
        if ns_variant not in ("legacy", "asym5") or (ns_variant == "asym5" and not 1 <= ns_steps <= 5):
            raise ValueError("Muon requires legacy, or asym5 with one to five Newton-Schulz steps.")
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_variant=ns_variant,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        muon_params = list(muon_params)
        params = muon_params.copy()
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim != 2:
                raise ValueError(f"Muon only supports 2D parameters, but got {p.ndim}D parameter (shape: {p.shape}).")
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr: float, param_shape: list[int]) -> float:
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon loop
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, group["ns_steps"], group.get("ns_variant", "legacy"))

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            # Adam backup
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss
