"""
MASE transform pass: swap the inner MHA of ``TimeSelfAttention`` with
``RoPEFusedMHA``.

The pass identifies every ``TimeSelfAttention`` node in the ``MaseGraph``,
checks that the inner ``self_attention`` is a standard ``MHA`` (i.e. has not
already been replaced), and swaps it with a ``RoPEFusedMHA`` instance that
fuses the RoPE rotation directly into the tiled attention kernel.

The outer ``TimeSelfAttention`` shell (layer norm, residual connection,
dropout) is left completely untouched.

Performance benefit
-------------------
The standard path materialises four intermediate ``(B, H, S, D)`` global
memory tensors (cos, sin, rot_q, rot_k) before the attention matmul.
``RoPEFusedMHA`` keeps the rotated Q/K tiles in registers during the Triton
kernel, reducing global memory traffic significantly for typical Chronos2
sequence lengths (64–512 patches).

On CPU or when Triton is unavailable the pass still applies: the eager
fallback path inside ``RoPEFusedMHA`` is slightly more efficient than the
original ``MHA`` path because cos/sin are computed once (rather than being
recomputed inside ``RoPE.forward`` which re-expands ``inv_freq`` on every
call).

Usage::

    from chop.passes.graph.transforms.timeseries import (
        fused_rope_time_attention_transform_pass,
    )

    mg, info = fused_rope_time_attention_transform_pass(mg)
    # No pass_args required — RoPE hyper-parameters are read from the model config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from chop.models.chronos2.layers import MHA, TimeSelfAttention
from chop.models.chronos2.optimized_layers import RoPEFusedMHA

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Model analyser
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class RoPETimeAttnInfo:
    """Analysis result for a single graph node."""

    node_name: str
    module_path: str
    can_optimise: bool
    reason: str


class RoPETimeAttnAnalyser:
    """Walk a ``MaseGraph`` and identify ``TimeSelfAttention`` nodes eligible
    for the fused RoPE + attention transform."""

    @staticmethod
    def analyse(mg) -> list[RoPETimeAttnInfo]:
        """Return analysis info for every ``TimeSelfAttention`` node.

        Eligible nodes are those whose inner ``self_attention`` is a plain
        ``MHA`` instance (i.e. not yet replaced with ``RoPEFusedMHA``).
        """
        results: list[RoPETimeAttnInfo] = []
        for node in mg.fx_graph.nodes:
            if node.op != "call_module":
                continue
            module = mg.model.get_submodule(node.target)
            if not isinstance(module, TimeSelfAttention):
                continue
            if isinstance(module.self_attention, RoPEFusedMHA):
                results.append(
                    RoPETimeAttnInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=False,
                        reason="Inner MHA already replaced with RoPEFusedMHA",
                    )
                )
            elif isinstance(module.self_attention, MHA) and module.self_attention.use_rope:
                results.append(
                    RoPETimeAttnInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=True,
                        reason="TimeSelfAttention with standard RoPE MHA detected",
                    )
                )
            else:
                results.append(
                    RoPETimeAttnInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=False,
                        reason="TimeSelfAttention inner MHA does not use RoPE — skipping",
                    )
                )
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Pass entry point
# ═══════════════════════════════════════════════════════════════════════════════
def fused_rope_time_attention_transform_pass(
    mg,
    pass_args: dict | None = None,
) -> tuple:
    """Replace the inner MHA of every ``TimeSelfAttention`` in *mg* with
    ``RoPEFusedMHA``.

    The outer ``TimeSelfAttention`` (layer norm, residual, dropout) is
    unchanged; only ``module.self_attention`` is swapped.

    Args:
        mg: ``MaseGraph`` wrapping a ``Chronos2Model``.
        pass_args:
            Currently unused; reserved for future options such as forcing
            a specific dispatch path.

    Returns:
        ``(mg, info_dict)`` where *info_dict* contains:

        - ``replaced``  (int): number of modules swapped.
        - ``analysis``  (list[RoPETimeAttnInfo]): per-node analysis results.
        - ``triton_active`` (bool): whether the Triton path will be used at
          runtime (depends on device and Triton availability).
    """
    if pass_args is None:
        pass_args = {}

    analysis = RoPETimeAttnAnalyser.analyse(mg)

    triton_active: bool | None = None
    replaced = 0

    for info in analysis:
        if not info.can_optimise:
            logger.debug("Skipping %s: %s", info.node_name, info.reason)
            continue

        module: TimeSelfAttention = mg.model.get_submodule(info.module_path)
        original_mha: MHA = module.self_attention
        device = next(module.parameters()).device

        fused_mha = RoPEFusedMHA(config=original_mha.config)

        # Transfer all learnable weights (q, k, v, o).
        # inv_freq is a non-persistent buffer — not in state_dict — and is
        # recomputed from the config inside RoPEFusedMHA.__init__.
        fused_mha.load_state_dict(original_mha.state_dict(), strict=False)
        fused_mha.to(device)

        # Swap the inner MHA — outer TimeSelfAttention is untouched
        module.self_attention = fused_mha

        if triton_active is None:
            triton_active = fused_mha._use_triton and device.type == "cuda"

        logger.info(
            "Replaced self_attention in %s with RoPEFusedMHA (triton=%s)",
            info.module_path,
            fused_mha._use_triton,
        )

        # Write metadata for downstream passes
        node = _find_node_by_target(mg, info.module_path)
        if node is not None and "mase" in node.meta:
            ts_meta = node.meta["mase"].parameters.setdefault("timeseries", {})
            ts_meta["rope_fused"] = True
            ts_meta["triton_active"] = fused_mha._use_triton

        replaced += 1

    return mg, {
        "replaced": replaced,
        "analysis": analysis,
        "triton_active": bool(triton_active) if triton_active is not None else False,
    }


def _find_node_by_target(mg, target: str):
    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and node.target == target:
            return node
    return None
