"""
MASE transform pass: swap the inner MHA of ``GroupSelfAttention`` with
``GroupAwareMHA``.

``group_ids`` must be provided in ``pass_args``.  The pass computes the group
partition once on CPU, writes it into each node's MASE metadata, and replaces
``group_self_attn.self_attention`` with a ``GroupAwareMHA`` instance so that
``forward()`` never inspects the mask at runtime.

The outer ``GroupSelfAttention`` shell (layer norm, residual connection,
dropout, axis transposes) is left completely untouched.

Usage::

    from chop.passes.graph.transforms.timeseries import (
        fast_group_attention_transform_pass,
    )

    mg, info = fast_group_attention_transform_pass(
        mg, pass_args={"group_ids": torch.arange(64)}
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from chop.models.chronos2.layers import GroupSelfAttention
from chop.models.chronos2.optimized_layers import (
    GroupAwareMHA,
    GroupPartition,
    KernelDispatcher,
    KernelVariant,
    UnivariateGroupAwareMHA,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Model analyser
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GroupAttentionInfo:
    """Analysis result for a single graph node."""

    node_name: str
    module_path: str
    can_optimise: bool
    reason: str


class GroupAttentionAnalyser:
    """Walk a ``MaseGraph`` and identify nodes eligible for the grouped
    sparse attention transform."""

    @staticmethod
    def analyse(mg) -> list[GroupAttentionInfo]:
        """Return analysis info for every ``call_module`` node.

        Eligible nodes are ``GroupSelfAttention`` instances whose inner MHA
        has not yet been replaced with ``GroupAwareMHA``.
        """
        results: list[GroupAttentionInfo] = []
        for node in mg.fx_graph.nodes:
            if node.op != "call_module":
                continue
            module = mg.model.get_submodule(node.target)
            if not isinstance(module, GroupSelfAttention):
                continue
            if isinstance(module.self_attention, (GroupAwareMHA, UnivariateGroupAwareMHA)):
                results.append(
                    GroupAttentionInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=False,
                        reason="Inner MHA already replaced with GroupAwareMHA or UnivariateGroupAwareMHA",
                    )
                )
            else:
                results.append(
                    GroupAttentionInfo(
                        node_name=node.name,
                        module_path=node.target,
                        can_optimise=True,
                        reason="GroupSelfAttention with standard MHA detected",
                    )
                )
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Pass entry point
# ═══════════════════════════════════════════════════════════════════════════════
def fast_group_attention_transform_pass(
    mg,
    pass_args: dict | None = None,
) -> tuple:
    """Replace the inner MHA of every ``GroupSelfAttention`` in *mg* with
    ``GroupAwareMHA``.

    The outer ``GroupSelfAttention`` (layer norm, residual, dropout) is
    unchanged; only ``module.self_attention`` is swapped.

    Args:
        mg: ``MaseGraph`` wrapping a Chronos2Model.
        pass_args:
            group_ids (torch.Tensor, required):
                1-D long tensor of shape ``(B,)`` matching the batch size that
                will be used at inference.
            kernel_variant (str, optional):
                Force a specific kernel: ``"univariate"``, ``"triton"``, or
                ``"packed_sparse"``. Defaults to auto-selection via
                ``KernelDispatcher``.

    Returns:
        ``(mg, info_dict)`` where *info_dict* contains ``replaced`` count,
        the ``partition``, and per-node ``analysis``.
    """
    if pass_args is None:
        pass_args = {}

    group_ids: torch.Tensor | None = pass_args.get("group_ids")
    if group_ids is None:
        raise ValueError(
            "fast_group_attention_transform_pass requires 'group_ids' in "
            "pass_args.  Example: pass_args={'group_ids': torch.arange(64)}"
        )

    partition = GroupPartition.from_group_ids(group_ids)

    forced_variant_name: str | None = pass_args.get("kernel_variant")
    forced_variant: KernelVariant | None = None
    if forced_variant_name is not None:
        forced_variant = KernelVariant[forced_variant_name.upper()]

    analysis = GroupAttentionAnalyser.analyse(mg)

    replaced = 0
    for info in analysis:
        if not info.can_optimise:
            logger.debug("Skipping %s: %s", info.node_name, info.reason)
            continue

        module: GroupSelfAttention = mg.model.get_submodule(info.module_path)
        device = next(module.parameters()).device

        if partition.all_univariate:
            # All groups are size 1: attention output is always V.
            # The pass handles this at compile time — no kernel dispatch needed.
            logger.info(
                "Replacing self_attention in %s with UnivariateGroupAwareMHA",
                info.module_path,
            )
            new_mha = UnivariateGroupAwareMHA(config=module.self_attention.config)
            mha_sd = module.self_attention.state_dict()
            new_mha.load_state_dict(
                {k: v for k, v in mha_sd.items() if k in ("v.weight", "o.weight")}
            )
            variant_name = "UNIVARIATE"
        else:
            variant = forced_variant or KernelDispatcher.select(partition, device)
            logger.info(
                "Replacing self_attention in %s with GroupAwareMHA (variant=%s)",
                info.module_path,
                variant.name,
            )
            new_mha = GroupAwareMHA(
                config=module.self_attention.config,
                partition=partition,
                variant=variant,
            )
            new_mha.load_state_dict(module.self_attention.state_dict())
            variant_name = variant.name

        new_mha.to(device)

        # Swap the inner MHA — outer GroupSelfAttention is untouched
        module.self_attention = new_mha

        # Write metadata for downstream passes
        node = _find_node_by_target(mg, info.module_path)
        if node is not None and "mase" in node.meta:
            ts_meta = node.meta["mase"].parameters.setdefault("timeseries", {})
            ts_meta["group_ids"] = group_ids.cpu()
            ts_meta["partition"] = partition
            ts_meta["kernel_variant"] = variant_name

        replaced += 1

    return mg, {
        "replaced": replaced,
        "partition": partition,
        "analysis": analysis,
    }


def _find_node_by_target(mg, target: str):
    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and node.target == target:
            return node
    return None
